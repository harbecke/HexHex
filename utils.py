def _one_pass(iters):
    for it in iters:
        try:
            yield next(it)
        except StopIteration:
            pass

def zip_list_of_lists_first_dim_reversed(*iterables):
    iters = [reversed(it) for it in iterables]
    output_list = []
    while True:
        iter_list = list(_one_pass(iters))
        output_list.extend(list(iter_list))
        if iter_list==[]:
            return output_list
