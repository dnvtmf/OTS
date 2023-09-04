import torch
from torch import Tensor
from tree_segmentation.extension._C import get_C_function, have_C_functions


class Mask:
    """like run-length encoding (RLE), but only encode last dim"""
    dtype = torch.uint8

    def __init__(self, data: Tensor) -> None:
        self.shape = data.shape
        self.start, self.counts = self._from_binary(data)

    def to_binary(self, engine='C'):
        if engine in ['auto', 'C'] and have_C_functions('mask_to_binary'):
            return get_C_function('mask_to_binary')(self.shape[-1], self.start, self.counts)
        # engine == python
        mask = torch.zeros(self.shape, dtype=torch.bool, device=self.start.device).flatten(0, -2)
        print(mask.shape)
        N = self.shape[-1]
        for i in range(len(self.start)):
            s = self.start[i]
            e = len(self.counts) if i + 1 == len(self.start) else self.start[i + 1]
            n = 0
            for j in range(s + 1, e, 2):
                n += self.counts[j - 1]
                mask[i, n:n + self.counts[j]] = 1
                n += self.counts[j]
        return mask.view(self.shape)

    def _from_binary(self, data: Tensor):
        N = data.shape[-1]
        data = torch.constant_pad_nd(data, (1, 1), 0)
        data = torch.constant_pad_nd(data, (1, 1), 1)
        change_index = torch.nonzero(data[..., :-1] != data[..., 1:])
        counts = change_index[1:, -1] - change_index[:-1, -1]
        start = torch.nonzero(counts < 0)[:, 0] - torch.arange(data[..., 0].numel() - 1, device=data.device)
        start = torch.constant_pad_nd(start, (1, 0), 0)
        counts = counts[counts >= 0]
        # assert start.max() < len(counts)
        counts[start] -= 1
        counts[start[1:] - 1] -= 1
        counts[-1] -= 1
        max_value = counts.max()
        if max_value < 256:
            counts = counts.to(torch.uint8)
        elif max_value < (1 << 15):
            counts = counts.to(torch.int16)
        else:
            counts = counts.to(torch.int32)
        start = start.view(data.shape[:-1]).to(torch.int32)
        return start, counts

    @classmethod
    def from_binary(cls, data: Tensor):
        return cls(data)

    @property
    def area(self):
        start = self.start.view(-1)
        nz = self.counts.clone()
        zero_index = torch.ones(len(self.counts), device=start.device)
        zero_index[start] = 0
        nz[torch.cumsum(zero_index, dim=0) % 2 == 0] = 0
        sum_nz = torch.cumsum(nz, dim=0)
        index = torch.constant_pad_nd(start, (0, 1), len(self.counts))
        return (sum_nz[index[1:] - 1] - sum_nz[index[:-1]]).sum(dim=-1).view(self.shape[:-2])

    def intersect(self, other: 'Mask', engine='auto'):
        """|self & other|"""
        H, W = self.shape[-2:]
        assert other.shape[-2:] == self.shape[-2:]
        area = torch.zeros(list(self.shape[:-2]) + list(other.shape[:-2]), dtype=torch.float, device=self.start.device)
        if engine in ['auto', 'C'] and have_C_functions('intersect'):
            get_C_function('intersect')(W, self.start, self.counts, other.start, other.counts, area)
            return area
        ## python engine
        print('Use python engine, please try C engine for speed')
        area_ = area.view(self.start[..., 0].numel(), -1)
        sa = self.start.view(-1, H)
        sb = other.start.view(-1, H)
        for i in range(area_.shape[0]):
            for j in range(area_.shape[1]):
                temp = 0
                for k in range(self.shape[-2]):
                    si, sj = sa[i, k].item(), sb[j, k].item()
                    la, lb = self.counts[si].item(), other.counts[sj].item()
                    na, nb = la, lb
                    a, b = 0, 0
                    while na < W or nb < W:
                        if na < nb:
                            a ^= 1
                            si += 1
                            la = self.counts[si].item()
                            na += la
                        else:
                            b ^= 1
                            sj += 1
                            lb = other.counts[sj].item()
                            nb += lb
                        if a == 1 and b == 1:
                            temp += min(na, nb) - max(na - la, nb - lb)
                area_[i, j] = temp
        return area

    def In(self, other: 'Mask'):
        """ |self & other| / |self|"""
        inter = self.intersect(other)
        area = self.area
        return (inter.view(area.numel(), -1) / area.view(-1, 1)).view_as(inter)

    def IoU(self, other: 'Mask'):
        inter = self.intersect(other)
        shape = inter.shape
        area = self.area.view(-1, 1)
        area_o = other.area.view(1, -1)
        inter = inter.view(area.numel(), -1)
        return (inter / (area + area_o - inter)).view(shape)

    def cpu(self):
        self.start = self.start.cpu()
        self.counts = self.counts.cpu()
        return self

    def cuda(self):
        self.start = self.start.cuda()
        self.counts = self.counts.cuda()
        return self

    def to(self, device=None):
        self.start = self.start.to(device=device)
        self.counts = self.counts.to(device=device)
        return self


class Masks:

    def __init__(self) -> None:
        pass


def test():
    a = torch.tensor(
        [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 1, 0, 1], [1, 0, 0, 1, 0], [1, 0, 0, 1, 1], [0, 1, 0, 1, 0]],
        dtype=torch.bool)
    print(a)
    print(a.shape)
    b = Mask(a)
    assert (b.to_binary(engine='python') == a).all()
    assert (b.to_binary(engine='C') == a).all()
    print('counts', b.counts)
    print('start', b.start)
    print(b.area.shape, b.area, a.sum())
    print('cpu:', b.intersect(b, engine='C'))
    b.cuda()
    assert (b.to_binary(engine='C').cpu() == a).all()
    print('cuda:', b.intersect(b, engine='C'))
    print('python:', b.intersect(b, engine='python'))
    print(b.In(b))
    print(b.IoU(b))


if __name__ == '__main__':
    test()