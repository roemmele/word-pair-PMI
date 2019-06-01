import pandas
import numpy
import seaborn
from scipy.special import logsumexp


def plot_pmi_distribution_graph(word_pair_scores):
    plot = seaborn.distplot(word_pair_scores, hist=True, kde=False,
                            bins=5, axlabel='PMI', label='# Pairs')
    return plot


def embed_prompts_in_dataframe(prompts, slice_idxs=(0, None)):
    dataframe = pandas.DataFrame({'Prompt': prompts}).iloc[slice_idxs[0]:slice_idxs[1]]
    if len(dataframe) == 1:
        dataframe = dataframe.rename({slice_idxs[0]: ''})
    return dataframe


def embed_stories_in_dataframe(stories_by_group_name, slice_idxs=(0, None)):
    dataframe = pandas.DataFrame(stories_by_group_name).iloc[slice_idxs[0]:slice_idxs[1]]
    if len(dataframe) == 1:
        dataframe = dataframe.rename({slice_idxs[0]: 'Story'}).T
    return dataframe


def embed_word_pair_scores_in_dataframe(word_pair_scores_by_group_name, slice_idxs=(0, None)):
    # import pdb
    # pdb.set_trace()
    dataframe = pandas.DataFrame(word_pair_scores_by_group_name).iloc[slice_idxs[0]:slice_idxs[1]]
    if len(dataframe) == 1:
        word_pair_scores_by_group_name = {group_name: word_pair_scores[slice_idxs[0]] for group_name, word_pair_scores
                                          in word_pair_scores_by_group_name.items()}
        display_n_word_pairs = min([len(word_pairs)
                                    for word_pairs in word_pair_scores_by_group_name.values()])
        word_pair_scores_by_group_name = {group_name: [(word_pair, numpy.round(score, 2)) for word_pair, score in word_pair_scores[:display_n_word_pairs]]
                                          for group_name, word_pair_scores in word_pair_scores_by_group_name.items()}
        dataframe = pandas.DataFrame(word_pair_scores_by_group_name)
        dataframe = dataframe.rename({idx: '' for idx in range(len(dataframe))})
    return dataframe


def embed_n_word_pairs_in_dataframe(n_word_pairs_by_group_name, slice_idxs=(0, None)):
    dataframe = pandas.DataFrame(n_word_pairs_by_group_name).iloc[slice_idxs[0]:slice_idxs[1]]
    return dataframe


def embed_word_pair_densities_in_dataframe(word_pair_densities_by_group_name, slice_idxs=(0, None)):
    dataframe = pandas.DataFrame(word_pair_densities_by_group_name).iloc[
        slice_idxs[0]:slice_idxs[1]]
    return dataframe


def embed_mean_pmi_sums_in_dataframe(mean_pmi_sums_by_group_name):
    dataframe = pandas.DataFrame([mean_pmi_sums_by_group_name]).rename({0: 'Mean'})
    return dataframe


def embed_mean_densities_in_dataframe(mean_word_pair_densities_by_group_name):
    dataframe = pandas.DataFrame([mean_word_pair_densities_by_group_name]).rename({0: 'Mean'})
    return dataframe


def embed_mean_n_word_pairs_in_dataframe(mean_n_word_pairs_by_group_name):
    dataframe = pandas.DataFrame([mean_n_word_pairs_by_group_name]).rename({0: 'Mean'})
    return dataframe


def embed_pmi_sums_in_dataframe(pmi_sums_by_group_name, slice_idxs=(0, None)):
    dataframe = pandas.DataFrame(pmi_sums_by_group_name).iloc[slice_idxs[0]:slice_idxs[1]]
    return dataframe


def embed_word_pairs_in_dataframe(pmi_model, n_pairs=None):
    dataframe = pandas.DataFrame([(word1, word2, count)
                                  for (word1, word2), count in pmi_model.pmi_scores.items()],
                                 columns=['word1', 'word2', 'PMI'])
    dataframe.set_index(['word1', 'word2'])
    dataframe = dataframe.sort_values(by='PMI', ascending=False)[:n_pairs]
    return dataframe


def get_prompts(filepath, n_items=None):
    prompts = [prompt.strip() for prompt in open(filepath)][:n_items]
    return prompts


def get_stories_by_group(filepaths_by_group_name, n_items=None):
    stories_by_group_name = {group_name: [story.strip() for story in open(filepath)][:n_items]
                             for group_name, filepath in filepaths_by_group_name.items()}
    return stories_by_group_name


# def get_word_pairs_with_pmi_for_stories(stories, pmi_model, tokenize_on_space=False):

#     word_pairs_by_story = [pmi_model.extract_word_pairs_in_text(text=story, tokenize_on_space=tokenize_on_space)
#                            for story in stories]
#     word_pairs_with_pmi_by_story = [pmi_model.get_pmi_scores_for_pairs(word_pairs)
#                                     for word_pairs in word_pairs_by_story]
#     return word_pairs_with_pmi_by_story


def get_n_word_pairs_for_stories_by_group(word_pair_scores_by_group):
    n_word_pairs_by_group = {}
    for group, group_word_pair_scores in word_pair_scores_by_group.items():
        n_word_pairs_by_group[group] = get_n_word_pairs_for_stories(group_word_pair_scores)
    return n_word_pairs_by_group


# def get_n_word_pairs_with_pmi_for_stories_by_group(stories_by_group_name, pmi_model):
#     word_pairs_by_group = get_word_pairs_for_stories_by_group(stories_by_group_name, pmi_model)
#     n_word_pairs_with_pmi_by_group = {}
#     for group_name, group_word_pairs in word_pairs_by_group.items():
#         n_word_pairs_with_pmi_by_group[group_name] = [len(pmi_model.get_pmi_scores_for_pairs(word_pairs))
#                                                       for word_pairs in group_word_pairs]
#     return n_word_pairs_with_pmi_by_group


def get_n_word_pairs_for_stories(word_pair_scores_by_story):

    n_word_pairs_by_story = [len(word_pair_scores)
                             for word_pair_scores in word_pair_scores_by_story]
    return n_word_pairs_by_story


def get_means_for_n_word_pairs_by_group(n_word_pairs_by_group_name):

    mean_n_word_pairs = {group_name: numpy.mean(group_n_word_pairs)
                         for group_name, group_n_word_pairs in n_word_pairs_by_group_name.items()}

    return mean_n_word_pairs


def get_means_for_n_word_pairs_with_pmi_by_group(n_word_pairs_with_pmi_by_group_name):

    mean_n_word_pairs_with_pmi = {group_name: numpy.mean(group_n_word_pairs_with_pmi)
                                  for group_name, group_n_word_pairs_with_pmi in n_word_pairs_with_pmi_by_group_name.items()}

    return mean_n_word_pairs_with_pmi


def get_word_pair_scores_for_stories(stories, pmi_model, tokenize_on_space=True):

    word_pairs_by_story = [pmi_model.extract_word_pairs_in_text(text=story, tokenize_on_space=tokenize_on_space)
                           for story in stories]
    word_pair_scores_by_story = [pmi_model.get_pmi_scores_for_pairs(word_pairs)
                                 for word_pairs in word_pairs_by_story]
    return word_pair_scores_by_story


def get_word_pair_scores_by_group(stories_by_group, pmi_model, tokenize_on_space=True):
    # import pdb
    # pdb.set_trace()
    word_pair_scores_by_group = {}
    for group, group_stories in stories_by_group.items():
        # if group in ('FAIR', 'L2W', 'Gold'):
        #     tokenize_on_space = True
        # else:
        #     tokenize_on_space = False
        word_pair_scores_by_group[group] = get_word_pair_scores_for_stories(group_stories,
                                                                            pmi_model, tokenize_on_space=tokenize_on_space)
        print("calculated word pair scores for all", group, "stories")
    return word_pair_scores_by_group


# def get_word_pairs_for_stories_by_group(stories_by_group_name, pmi_model, tokenize_on_space=False):
#     # import pdb
#     # pdb.set_trace()
#     word_pairs_by_group = {}
#     for group_name, group_stories in stories_by_group_name.items():
#         word_pairs_by_group[group_name] = [pmi_model.extract_word_pairs_in_text(text=story, tokenize_on_space=tokenize_on_space)
#                                            for story in group_stories]
#     return word_pairs_by_group


def get_word_pairs_density(word_pairs, word_pair_scores):
    density = len(word_pair_scores) / len(word_pairs)
    return density


def get_word_pair_densities_for_stories_by_group(stories_by_group_name, pmi_model, tokenize_on_space=True):
    word_pair_densities_by_group = {}

    for group_name, group_stories in stories_by_group_name.items():
        group_word_pairs = [pmi_model.extract_word_pairs_in_text(text=story, tokenize_on_space=tokenize_on_space)
                            for story in group_stories]
        group_word_pair_scores = [pmi_model.get_pmi_scores_for_pairs(
            word_pairs) for word_pairs in group_word_pairs]
        group_densities = [get_word_pairs_density(word_pairs, word_pair_scores)
                           for word_pairs, word_pair_scores in zip(group_word_pairs, group_word_pair_scores)]
        word_pair_densities_by_group[group_name] = group_densities

    return word_pair_densities_by_group


def get_means_for_densities_by_group(word_pair_densities_by_group_name):

    mean_densities = {group_name: numpy.mean(group_densities)
                      for group_name, group_densities in word_pair_densities_by_group_name.items()}

    return mean_densities


def get_pmi_sums_for_word_pairs(word_pair_scores):

    if not word_pair_scores:
        sum_pmi_score = numpy.log(1e-10)
    else:
        # - numpy.log(len(word_pair_scores))
        sum_pmi_score = logsumexp([score for word_pair, score in word_pair_scores])
    return sum_pmi_score


def get_pmi_sums_for_stories(word_pair_scores_by_story, top_k_pairs=None):

    pmi_sums = []
    for idx, word_pair_scores in enumerate(word_pair_scores_by_story):
        # word_pairs = pmi_model.extract_word_pairs_in_text(text=story,
        #                                                   tokenize_on_space=tokenize_on_space)
        # word_pair_scores = pmi_model.get_pmi_scores_for_pairs(word_pairs)[:top_k_pairs]
        #word_pairs_density = get_word_pairs_density(word_pairs, word_pair_scores)
        pmi_sum = get_pmi_sums_for_word_pairs(word_pair_scores)
        # pmi_sum = pmi_sum + word_pairs_density  # Scale by density
        pmi_sums.append(pmi_sum)
        if idx and idx % 2000 == 0:
            print("calculated pmi sums for", idx, "stories...")
    return pmi_sums


def get_pmi_sums_for_stories_by_group(word_pair_scores_by_group, top_k_pairs=None):

    pmi_sums_by_group = {}
    for group, group_word_pair_scores in word_pair_scores_by_group.items():
        pmi_sums = get_pmi_sums_for_stories(group_word_pair_scores, top_k_pairs=top_k_pairs)
        pmi_sums_by_group[group] = pmi_sums
        print("finished calculating pmi sums for all", group, "stories")
    return pmi_sums_by_group


def get_means_for_pmi_sums_by_group(pmi_sums_by_group_name):

    mean_pmi_sums = {group_name: numpy.mean(group_pmi_sums)
                     for group_name, group_pmi_sums in pmi_sums_by_group_name.items()}

    return mean_pmi_sums


def filter_word_pair_scores_by_top_k(word_pair_scores_by_group, top_k=None):

    # import pdb
    # pdb.set_trace()
    filtered_word_pair_scores_by_group = {}
    n_word_pairs_for_stories_by_group = get_n_word_pairs_for_stories_by_group(
        word_pair_scores_by_group)
    if not top_k:
        filtered_n_word_pairs_by_story = numpy.min(numpy.stack(n_word_pairs_for_stories_by_group.values(), axis=0),
                                                   axis=0)
    else:
        filtered_n_word_pairs_by_story = numpy.repeat(top_k, repeats=len(
            list(n_word_pairs_for_stories_by_group.values())[0]))

    for group, group_word_pair_scores in word_pair_scores_by_group.items():
        filtered_word_pair_scores_by_group[group] = [word_pair_scores[:n_word_pairs] for word_pair_scores, n_word_pairs
                                                     in zip(group_word_pair_scores, filtered_n_word_pairs_by_story)]
    return filtered_word_pair_scores_by_group

    # word_pair_scores_by_group_name = {group_name: word_pair_scores[slice_idxs[0]] for group_name, word_pair_scores
    #                                   in word_pair_scores_by_group_name.items()}
    # filtered_n_word_pairs = get_n_word_pairs_for_stories_by_group(word_pair_counts_by_group_name)
    # display_n_word_pairs = min([len(word_pairs)
    #                             for word_pairs in word_pair_scores_by_group_name.values()])
    # word_pair_scores_by_group_name = {group_name: [(word_pair, numpy.round(score, 3)) for word_pair, score in word_pair_scores[:display_n_word_pairs]]
    #                                   for group_name, word_pair_scores in word_pair_scores_by_group_name.items()}
    # dataframe = pandas.DataFrame(word_pair_scores_by_group_name)
    # dataframe = dataframe.rename({idx: '' for idx in range(len(dataframe))})
