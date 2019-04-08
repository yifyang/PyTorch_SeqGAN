import argparse
import os
import pickle as pkl
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from data_iter import DisDataIter, GenDataIter, prepare_dataloaders
from generator import Generator
from discriminator import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout
from loss import PGLoss
from transformer.Optim import ScheduledOptim
from transformer import Constants
from transformer.Models import Transformer


# Arguemnts
parser = argparse.ArgumentParser(description='Attention_SeqGAN')
parser.add_argument('--hpc', action='store_true', default=True,
                    help='set to hpc mode')
parser.add_argument('--data_path', type=str, default='dataset/', metavar='PATH',
                    help='data path to save files (default: dataset/)')
parser.add_argument('--rounds', type=int, default=200, metavar='N',
                    help='rounds of adversarial training (default: 150)')
parser.add_argument('--g_pretrain_steps', type=int, default=200, metavar='N',
                    help='steps of pre-training of generators (default: 120)')
parser.add_argument('--d_pretrain_steps', type=int, default=70, metavar='N',
                    help='steps of pre-training of discriminators (default: 50)')
parser.add_argument('--g_steps', type=int, default=1, metavar='N',
                    help='steps of generator updates in one round of adverarial training (default: 1)')
parser.add_argument('--d_steps', type=int, default=2, metavar='N',
                    help='steps of discriminator updates in one round of adverarial training (default: 3)')
parser.add_argument('--gk_epochs', type=int, default=2, metavar='N',
                    help='epochs of generator updates in one step of generate update (default: 1)')
parser.add_argument('--dk_epochs', type=int, default=2, metavar='N',
                    help='epochs of discriminator updates in one step of discriminator update (default: 3)')
parser.add_argument('--update_rate', type=float, default=0.8, metavar='UR',
                    help='update rate of roll-out model (default: 0.8)')
parser.add_argument('--n_rollout', type=int, default=16, metavar='N',
                    help='number of roll-out (default: 16)')
parser.add_argument('--vocab_size', type=int, default=28261, metavar='N',
                    help='vocabulary size (default: 28261)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--n_samples', type=int, default=35094, metavar='N',
                    help='number of samples gerenated per time (default: 35094)')
parser.add_argument('--gen_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of generator optimizer (default: 1e-3)')
parser.add_argument('--dis_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of discriminator optimizer (default: 1e-3)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--seq_len', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')


# Files
POSITIVE_FILE = 'news.data'
NEGATIVE_FILE = 'gen_news.data'
# RANDOM_FILE = 'self_num_rand.data'
EPOCH_FILE = 'epoch_self.data' # store samples every epoch during adversarial training


# Genrator Parameters
g_embed_dim = 512
g_hidden_dim = 32
# g_hidden_layer = 3
g_seq_len = 10


# Discriminator Parameters
d_num_class = 2
d_embed_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
# d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
# d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100]
d_dropout_prob = 0.2


def generate_samples(model, data_iter, args, output_file, ad_train=False, epoch_file=''):
    samples = []
    # for  in range(int(generated_num / batch_size)):
    #     sample = model.sample(batch_size, g_seq_len).cpu().data.numpy().tolist()
    #     samples.extend(sample)

    for batch in data_iter:
        if args.cuda:
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.cuda(), batch)
        else:
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.cpu(), batch)

        sample = model.sample(tgt_seq, tgt_pos, len(tgt_seq), args.seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)

    with open(output_file, 'w') as fout:
        for sample in samples:
            # string = ''.join([str(s) for s in sample])
            # fout.write('{}\n'.format(string))
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
    if ad_train:
        with open(epoch_file, 'a') as fout:
            for i, sample in enumerate(samples):
                if i > 9: break
                # string = ''.join([str(s) for s in sample])
                # fout.write('{}\n'.format(string))
                string = ' '.join([str(s) for s in sample])
                fout.write('%s\n' % string)


def cal_performance(pred, gold, critireon, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, critireon, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, critireon, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        # non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        # loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        log_prb = F.log_softmax(pred, dim=1)
        loss = critireon(log_prb, gold)

    return loss

def train_generator_MLE(gen, data_iter, criterion, optimizer, epochs, 
        gen_pretrain_train_loss, args):
    """
    Train generator with MLE
    """
    for epoch in range(epochs):
        total_loss = 0.
        for batch in data_iter:
            if args.cuda:
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.cuda(), batch)
            else:
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.cpu(), batch)
            # gold = tgt_seq[:, :-1]
            optimizer.zero_grad()

            # if args.cuda:
            #     data, target = data.cuda(), target.cuda()
            # target = target.contiguous().view(-1)
            output = gen(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(output, tgt_seq[:, :-1], criterion)
            loss.backward()
            optimizer.step_and_update_lr()
            total_loss += loss.item()
        # data_iter.reset()
    avg_loss = total_loss / len(batch)
    print("Epoch {}, train loss: {:.5f}".format(epoch, avg_loss))
    gen_pretrain_train_loss.append(avg_loss)


def train_generator_PG(gen, dis, gen_data_iter, rollout, pg_loss, optimizer, epochs, args):
    """
    Train generator with the guidance of policy gradient
    """
    sample_batch = gen_data_iter.__iter__().__next__()
    if args.cuda:
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.cuda(), sample_batch)
    else:
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.cpu(), sample_batch)

    for epoch in range(epochs):
        # construct the input to the genrator, add zeros before samples and delete the last column
        # model.sample(tgt_seq, tgt_pos, len(tgt_seq), seq_len)

        samples = generator.sample(tgt_seq, tgt_pos, len(tgt_seq), args.seq_len)
        zeros = torch.zeros(args.batch_size, 1, dtype=torch.int64)
        if samples.is_cuda:
            zeros = zeros.cuda()
        inputs = torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous()
        targets = samples.data.contiguous().view((-1,))

        # calculate the reward
        rewards = torch.tensor(rollout.get_reward(samples, tgt_seq, tgt_pos, args.n_rollout, dis))
        if args.cuda:
            rewards = rewards.cuda()

        # update generator
        output = gen(inputs)
        loss = pg_loss(output, targets, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch {}, train loss: {:.5f}".format(epoch, loss))


def eval_generator(model, data_iter, criterion, args):
    """
    Evaluate generator with NLL
    """
    total_loss = 0.
    with torch.no_grad():
        for batch in data_iter:
            # if args.cuda:
            #     data, target = data.cuda(), target.cuda()
            if args.cuda:
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.cuda(), batch)
            else:
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.cpu(), batch)
            target = target.contiguous().view(-1)
            pred = model(src_seq)
            loss = criterion(pred, tgt_seq)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_iter)
    return avg_loss


def train_discriminator(dis, gen, gen_data_iter, criterion, optimizer, epochs,
        dis_adversarial_train_loss, dis_adversarial_train_acc, args):
    """
    Train discriminator
    """
    generate_samples(gen, gen_data_iter, args, NEGATIVE_FILE)
    # generate_samples(gen, args.batch_size, args.n_samples, NEGATIVE_FILE)
    data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, args.batch_size)
    for epoch in range(epochs):
        correct = 0
        total_loss = 0.
        for data, target in data_iter:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            output = dis(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            loss = criterion(output, target)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        data_iter.reset()
        avg_loss = total_loss / len(data_iter)
        acc = correct.item() / data_iter.data_num
        print("Epoch {}, train loss: {:.5f}, train acc: {:.3f}".format(epoch, avg_loss, acc))
        dis_adversarial_train_loss.append(avg_loss)
        dis_adversarial_train_acc.append(acc)


def eval_discriminator(model, data_iter, criterion, args):
    """
    Evaluate discriminator, dropout is enabled
    """
    correct = 0
    total_loss = 0.
    with torch.no_grad():
        for data, target in data_iter:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            output = model(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_iter)
    acc = correct.item() / data_iter.data_num
    return avg_loss, acc


def adversarial_train(gen, dis, gen_data_iter, rollout, pg_loss, nll_loss, gen_optimizer, dis_optimizer,
        dis_adversarial_train_loss, dis_adversarial_train_acc, args):
    """
    Adversarially train generator and discriminator
    """
    # train generator for g_steps
    print("#Train generator")
    for i in range(args.g_steps):
        print("##G-Step {}".format(i))
        train_generator_PG(gen, dis, gen_data_iter, rollout, pg_loss, gen_optimizer, args.gk_epochs, args)

    # train discriminator for d_steps
    print("#Train discriminator")
    for i in range(args.d_steps):
        print("##D-Step {}".format(i))
        train_discriminator(dis, gen, gen_data_iter, nll_loss, dis_optimizer, args.dk_epochs,
            dis_adversarial_train_loss, dis_adversarial_train_acc, args)

    # update roll-out model
    rollout.update_params()


if __name__ == '__main__':
    # Parse arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if not args.hpc:
        args.data_path = 'data_test/'
    POSITIVE_FILE = args.data_path + POSITIVE_FILE
    NEGATIVE_FILE = args.data_path + NEGATIVE_FILE
    EPOCH_FILE = args.data_path + EPOCH_FILE
    # RANDOM_FILE = args.data_path + RANDOM_FILE
    if os.path.exists(EPOCH_FILE):
        os.remove(EPOCH_FILE)

    # Set models, criteria, optimizers
    # generator = Generator(args.vocab_size, g_embed_dim, g_hidden_dim, args.cuda)

    generator = Transformer(args.vocab_size, args.vocab_size, args.seq_len+1)
    discriminator = Discriminator(d_num_class, args.vocab_size, d_embed_dim, d_filter_sizes, d_num_filters, d_dropout_prob)
    target_lstm = TargetLSTM(args.vocab_size, g_embed_dim, g_hidden_dim, args.cuda)
    nll_loss = nn.NLLLoss()
    pg_loss = PGLoss()
    if args.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        target_lstm = target_lstm.cuda()
        nll_loss = nll_loss.cuda()
        pg_loss = pg_loss.cuda()
        cudnn.benchmark = True
    # gen_optimizer = optim.Adam(params=generator.parameters(), lr=args.gen_lr)
    gen_optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, generator.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        512, 4000)
    dis_optimizer = optim.SGD(params=discriminator.parameters(), lr=args.dis_lr)

    # Container of experiment data
    gen_pretrain_train_loss = []
    gen_pretrain_eval_loss = []
    dis_pretrain_train_loss = []
    dis_pretrain_train_acc = []
    dis_pretrain_eval_loss = []
    dis_pretrain_eval_acc = []
    gen_adversarial_eval_loss = []
    dis_adversarial_train_loss = []
    dis_adversarial_train_acc = []
    dis_adversarial_eval_loss = []
    dis_adversarial_eval_acc = []

    # Generate toy data using target LSTM
    # print('#####################################################')
    # print('Generating data ...')
    # print('#####################################################\n\n')
    # generate_samples(target_lstm, args.batch_size, args.n_samples, POSITIVE_FILE)

    # Pre-train generator using MLE
    print('#####################################################')
    print('Start pre-training generator with MLE...')
    print('#####################################################\n')

    # gen_data_iter = GenDataIter(POSITIVE_FILE, args.batch_size)
    gen_data_iter = prepare_dataloaders(POSITIVE_FILE, args.batch_size)

    for i in range(args.g_pretrain_steps):
        print("G-Step {}".format(i))
        # train_generator_MLE(generator, gen_data_iter, nll_loss,
        #     gen_optimizer, args.gk_epochs, gen_pretrain_train_loss, args)
        generate_samples(generator, gen_data_iter, args, NEGATIVE_FILE)
        eval_iter = prepare_dataloaders(NEGATIVE_FILE, args.batch_size)
        gen_loss = eval_generator(target_lstm, eval_iter, nll_loss, args)
        gen_pretrain_eval_loss.append(gen_loss)
        print("eval loss: {:.5f}\n".format(gen_loss))
    print('#####################################################\n\n')

    # Pre-train discriminator
    print('#####################################################')
    print('Start pre-training discriminator...')
    print('#####################################################\n')
    for i in range(args.d_pretrain_steps):
        print("D-Step {}".format(i))
        train_discriminator(discriminator, generator, gen_data_iter, nll_loss,
            dis_optimizer, args.dk_epochs, dis_adversarial_train_loss, dis_adversarial_train_acc, args)
        generate_samples(generator, gen_data_iter, args, NEGATIVE_FILE)
        # generate_samples(generator, args.batch_size, args.n_samples, NEGATIVE_FILE)
        eval_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, args.batch_size)
        dis_loss, dis_acc = eval_discriminator(discriminator, eval_iter, nll_loss, args)
        dis_pretrain_eval_loss.append(dis_loss)
        dis_pretrain_eval_acc.append(dis_acc)
        print("eval loss: {:.5f}, eval acc: {:.3f}\n".format(dis_loss, dis_acc))
    print('#####################################################\n\n')

    # Adversarial training
    print('#####################################################')
    print('Start adversarial training...')
    print('#####################################################\n')
    rollout = Rollout(generator, args.update_rate)
    for i in range(args.rounds):
        print("Round {}".format(i))
        adversarial_train(generator, discriminator, gen_data_iter, rollout,
            pg_loss, nll_loss, gen_optimizer, dis_optimizer, 
            dis_adversarial_train_loss, dis_adversarial_train_acc, args)

        # generate_samples(generator, args.batch_size, args.n_samples, NEGATIVE_FILE, ad_train=True, epoch_file=EPOCH_FILE)
        generate_samples(generator, gen_data_iter, args, NEGATIVE_FILE)

        gen_eval_iter = prepare_dataloaders(NEGATIVE_FILE, args.batch_size)
        dis_eval_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, args.batch_size)
        gen_loss = eval_generator(target_lstm, gen_eval_iter, nll_loss, args)
        gen_adversarial_eval_loss.append(gen_loss)
        dis_loss, dis_acc = eval_discriminator(discriminator, dis_eval_iter, nll_loss, args)
        dis_adversarial_eval_loss.append(dis_loss)
        dis_adversarial_eval_acc.append(dis_acc)
        print("gen eval loss: {:.5f}, dis eval loss: {:.5f}, dis eval acc: {:.3f}\n"
            .format(gen_loss, dis_loss, dis_acc))
        print("dis eval loss: {:.5f}, dis eval acc: {:.3f}\n"
              .format(dis_loss, dis_acc))

    """
    # Save experiment data
    with open(args.data_path + 'experiment.pkl', 'wb') as f:
        pkl.dump(
            (gen_pretrain_train_loss,
                gen_pretrain_eval_loss,
                dis_pretrain_train_loss,
                dis_pretrain_train_acc,
                dis_pretrain_eval_loss,
                dis_pretrain_eval_acc,
                gen_adversarial_eval_loss,
                dis_adversarial_train_loss,
                dis_adversarial_train_acc,
                dis_adversarial_eval_loss,
                dis_adversarial_eval_acc),
            f,
            protocol=pkl.HIGHEST_PROTOCOL
        )
    """
