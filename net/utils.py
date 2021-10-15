class Poly_LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, iters_per_epoch, base_poly=0.9, quiet=False):
        self.epoch = -1
        self.quiet = quiet
        self.base_lr = base_lr
        self.base_poly = base_poly
        self.iters_per_epoch = iters_per_epoch
        self.total_iters = num_epochs * iters_per_epoch

    def __call__(self, optimizer, writer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        lr = self.base_lr * pow((1 - 1.0 * T / self.total_iters), self.base_poly)
        if (not self.quiet) and (epoch != self.epoch):
            self.epoch = epoch
            writer.add_scalar('lr', lr, epoch)
            print('=> Epoch %i, lr = %.4f, best_pred = %.2f%s' % (epoch, lr, best_pred, '%'))
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr