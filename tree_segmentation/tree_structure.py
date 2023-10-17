import torch


class TreeStructure:

    def __init__(self, N=1, device=None, verbose=0):
        self.device = device
        self.parent = torch.empty(N, dtype=torch.int, device=device)
        self.first = torch.empty(N, dtype=torch.int, device=device)
        self.last = torch.empty(N, dtype=torch.int, device=device)
        self.next = torch.empty(N, dtype=torch.int, device=device)
        self.cnt = 0
        self.verbose = verbose
        self.reset()

    def reset(self):
        if self.verbose > 0:
            print('[Tree] reset')
        self.cnt = 0
        self.parent.fill_(-1)
        self.first.fill_(-1)
        self.last.fill_(-1)
        self.next.fill_(-1)

    def resize(self, N: int):
        M = self.parent.numel()
        if self.verbose > 0:
            print(f'[Tree] Resize tree from {M} to {N}')
        if M == N:
            return N, M
        if M > N:
            assert self.cnt <= N
            self.parent = self.parent[:N]
            self.first = self.first[:N]
            self.last = self.last[:N]
            self.next = self.next[:N]
            return N, M
        size = (N - M,)
        self.parent = torch.cat([self.parent, self.parent.new_full(size, -1)])
        self.first = torch.cat([self.first, self.first.new_full(size, -1)])
        self.last = torch.cat([self.last, self.last.new_full(size, -1)])
        self.next = torch.cat([self.next, self.next.new_full(size, -1)])
        return N, M

    def __len__(self):
        return self.cnt

    def node_new(self):
        assert self.cnt <= self.parent.shape[0]
        self.cnt += 1
        self.first[self.cnt] = -1
        self.last[self.cnt] = -1
        self.next[self.cnt] = -1
        self.parent[self.cnt] = -1
        if self.verbose > 1:
            print(f'[Tree] new node: {self.cnt}')
        return self.cnt

    def node_delete(self, idx: int, move_children=False):
        if self.verbose > 1:
            print(f'[Tree] delete node: {idx}, move_children={move_children}')
        last, next = self.last[idx].item(), self.next[idx].item()
        if last >= 0:
            self.next[last] = next
            if next >= 0:
                self.last[next] = last
        elif next >= 0:
            self.last[next] = last
        self.last[idx] = -1
        self.next[idx] = -1
        pa = self.parent[idx]
        if self.first[pa] == idx:
            self.first[pa] = last if last >= 0 else next
        if move_children:
            for child in self.get_children(idx):
                self.node_insert(child, pa)
            self.first[idx] = -1

    def node_replace(self, idx: int, old: int):
        if self.verbose > 1:
            print(f'[Tree] replace node: {old} by {idx}')
        last, next = self.last[old].item(), self.next[old].item()
        self.last[idx] = last
        if last >= 0:
            self.next[last] = idx
        self.next[idx] = next
        if next >= 0:
            self.last[next] = idx
        self.parent[idx] = self.parent[old]
        if self.first[self.parent[old]].item() == old:
            self.first[self.parent[old]] = idx
        if self.first[old].item() == -1:
            self.first[idx] = -1
            return
        children = self.get_children(old)
        self.first[idx] = children[0]
        for child in children:
            self.parent[child] = idx

    def node_insert(self, idx: int, parent: int):
        if self.verbose > 1:
            print(f'[Tree] insert node: {idx} below {parent}')
        self.parent[idx] = parent
        # self.first[idx] = -1
        next = self.first[parent].item()
        self.first[parent] = idx
        if next == -1:
            self.last[idx] = -1
            self.next[idx] = -1
        else:
            last = self.last[next].item()
            self.last[idx] = last
            self.next[idx] = next
            self.last[next] = idx
            if last >= 0:
                self.next[last] = idx

    def node_move(self, i: int, parent: int):
        if self.verbose > 1:
            print(f'[Tree] move node: {i} to the child of {parent} ')
        self.node_delete(i, move_children=False)
        self.node_insert(i, parent)

    def get_children(self, root: int):
        children = []
        child = self.first[root].item()
        if child == -1:
            return children
        while child != -1:
            children.append(child)
            child = self.next[child].item()
        child = self.last[children[0]]
        while child != -1:
            children.append(child)
            child = self.last[child].item()
        return children

    def get_level(self, root=0, level=1):
        assert level >= 0
        levels = self.get_levels(root, level)
        return levels[level] if len(levels) > level else torch.tensor([], dtype=torch.int, device=self.device)

    def get_levels(self, root=0, depth=-1):
        now_level = [root]
        levels = [torch.tensor(now_level, dtype=torch.int, device=self.device)]
        while len(now_level) > 0 and depth != 0:
            next_level = []
            for x in now_level:
                next_level.extend(self.get_children(x))
            if len(next_level) == 0:
                break
            levels.append(torch.tensor(next_level, dtype=torch.int, device=self.device))
            now_level = next_level
            depth -= 1

        return levels

    def get_depth(self, i: int):
        depth = 0
        while i != 0:
            if i < 0:
                return -1
            depth += 1
            i = self.parent[i].item()
        return depth

    def node_rearrange(self, indices=None):
        if self.verbose > 0:
            print(f'[Tree] rerange nodes')
        if indices is None:
            indices = torch.cat(self.get_levels(), dim=0)
        indices = indices.long()
        num = len(indices)
        assert 0 <= indices.min() and indices.max() <= self.cnt
        new_indices = indices.new_full((self.cnt + 2,), -1)
        new_indices[indices + 1] = torch.arange(len(indices), dtype=indices.dtype, device=indices.device)
        assert new_indices[1] == 0  # keep root is unchanged
        self.parent[:num] = new_indices[self.parent[indices].long() + 1]
        self.first[:num] = new_indices[self.first[indices].long() + 1]
        self.last[:num] = new_indices[self.last[indices].long() + 1]
        self.next[:num] = new_indices[self.next[indices].long() + 1]
        self.cnt = len(indices) - 1
        return indices, new_indices

    def print_tree(self):
        from rich.tree import Tree
        import rich
        levels = self.get_levels()
        print_tree = Tree('0: Tree Root')
        nodes = {0: print_tree}
        for i, level in enumerate(levels):
            if i == 0:
                continue
            for j in level:
                j = j.item()
                p = self.parent[j].item()
                nodes[j] = nodes[p].add(f"{j}")
        rich.print(print_tree)

    def to(self, device):
        self.device = device
        self.parent = self.parent.to(device)
        self.next = self.next.to(device)
        self.last = self.last.to(device)
        self.first = self.first.to(device)
        return self
