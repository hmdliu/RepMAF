
import random

class Poly_LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, iters_per_epoch,
                    base_poly=0.9, thresh=1e-6, quiet=False):
        self.epoch = -1
        self.quiet = quiet
        self.thresh = thresh
        self.curr_lr = base_lr
        self.base_lr = base_lr
        self.base_poly = base_poly
        self.iters_per_epoch = iters_per_epoch
        self.total_iters = num_epochs * iters_per_epoch

    def __call__(self, optimizer, writer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.curr_lr > self.thresh:
            lr = self.base_lr * pow((1 - 1.0 * T / self.total_iters), self.base_poly)
            self._adjust_learning_rate(optimizer, lr)
            self.curr_lr = lr
        if (not self.quiet) and (epoch != self.epoch):
            self.epoch = epoch
            writer.add_scalar('lr', self.curr_lr, epoch)
            print('=> Epoch %i, lr = %.4f, best_pred = %.2f%s' % (epoch, self.curr_lr, best_pred, '%'))

    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr

def get_fwd_branch(branch_num, branch_dropout, training_flag):
    if not training_flag:
        return [True for i in range(branch_num)]
    fwd = [(random.random() > branch_dropout) for i in range(branch_num)]
    if sum(fwd) == 0:
        fwd[random.randint(0, branch_num-1)] = True
    return fwd

if __name__ == '__main__':
    for i in range(10):
        print(get_fwd_branch(4, 0.2))
