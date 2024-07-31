import math
import random
import attr
from typing import Optional
from datasets import Dataset
from prompts.base import ExampleTemplate
from tools.param_impl import Parameters
from constants import ExSel as ES

@attr.s(auto_attribs=True)
class CommonSelectorArgs(Parameters):
    selector_type: ES
    n_shots: int

@attr.s(auto_attribs=True)
class StructuralSelectorArgs(CommonSelectorArgs):
    substruct: str = 'depst'
    subst_size: int = 4
    depparser: str = "spacy"

    def get_name(self):
        name_parts = [f'{self.subst_size}_subst']
        if self.substruct != 'depst':
            name_parts.append(self.substruct)
        elif self.depparser != 'spacy':
            name_parts.append(self.depparser)
        return '-'.join(name_parts)

@attr.s(auto_attribs=True)
class SamplingSelectorArgs(CommonSelectorArgs):
    n_shots: int = 4
    seed: int = 0
    n_combs: Optional[int] = 10000

def bag_relevance(target_bag, pred_bag, metric) -> float:
    common = pred_bag & target_bag
    if len(common) == 0: return 0
    recall = len(common) / len(target_bag)
    if metric == 'recall': return recall
    precision = len(common) / len(pred_bag)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def get_combinations(N, k, n_combs, seed):
    import itertools
    if n_combs == -1:
        combinations = itertools.combinations(range(N), k)
    else:
        assert 0 < n_combs < math.comb(N, k)
        random.seed(seed)
        def sample_combinations(choices, size, count):
            collected = {tuple(random.sample(choices, size)) for _ in range(count)}
            while len(collected) < count:
                collected.add(tuple(random.sample(choices, size)))
            return list(collected)
        combinations = sample_combinations(range(N), k, n_combs)
    return combinations

class SelectorUtilsMixin:
    @staticmethod
    def drop_duplicates(examples: Dataset, example_template: ExampleTemplate):
        unique_examples_idxes = dict()
        # breakpoint()
        for i, ex in enumerate(examples):
            try:
                s = example_template.format(**ex)
            except Exception:
                s = example_template.format(**ex, test=True)
            if s not in unique_examples_idxes:
                unique_examples_idxes[s] = i
        #     else:
        #         print(f"Duplicate found: {s}")
        #         print(f"Original example: {examples[unique_examples_idxes[s]]}")
        #         print(f"Duplicate example: {ex}")
        #         print("---")
        # print(f"Original count: {len(examples)}, After deduplication: {len(unique_examples_idxes)}")
    
        return examples.select(list(unique_examples_idxes.values()))

    @staticmethod
    def get_substructs(strings, args: StructuralSelectorArgs, parser=None, verbose=False):
        from tools.structure.substructs import get_substructs
        return get_substructs(strings, args.substruct, args.subst_size, args.depparser, parser, verbose)

    @staticmethod
    def get_combinations(N, args: SamplingSelectorArgs, cand2struct=None):
        combinations = get_combinations(
            N=N, k=args.n_shots, n_combs=args.n_combs, seed=args.seed)
        if combination2struct is not None:
            combination2struct = {
                comb: set([x for idx in comb for x in cand2struct[idx]])
                for comb in combinations}
            return combinations, combination2struct
        else:
            return combinations

    @classmethod
    def common_setup_1(cls, examples, query_examples, example_template):
        examples = cls.drop_duplicates(examples, example_template)
        query_examples = query_examples or []
        ex_to_string = lambda ex: example_template.format(**ex, test=True)
        cand_strings = [ex_to_string(ex) for ex in examples]
        query_strings = [ex_to_string(ex) for ex in query_examples]
        query2idx = {query: i for i, query in enumerate(query_strings)}
        return examples, query_examples, cand_strings, query_strings, query2idx

    @classmethod
    def common_setup_2(cls, examples, query_examples, ex_len_fn, max_len):
        max_len = max_len if max_len > 0 else 1000000
        get_len = lambda ex, **kwargs: ex_len_fn(ex, **kwargs) if ex_len_fn else 0
        cand_lens = [get_len(ex, test=False) for ex in examples]
        query_lens = [get_len(ex, test=True) for ex in query_examples]
        completed_query_lens = [get_len(ex, test=False) for ex in query_examples]

        return cand_lens, query_lens, completed_query_lens, max_len
