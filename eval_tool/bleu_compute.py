from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

def calculate_bleu_scores(reference_strs, candidate_str):
    if isinstance(reference_strs, str):
        references = [reference_strs.split()]
    elif isinstance(reference_strs, list):
        references = [ref.split() for ref in reference_strs]
    candidate = candidate_str.split()
    bleu_score = sentence_bleu(references, candidate)
    bleu_1gram = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
    bleu_2gram = sentence_bleu(references, candidate, weights=(0, 1, 0, 0))

    return bleu_score, bleu_1gram, bleu_2gram

def calculate_meteor_score(reference_strs, candidate_str):
    """
    Compute METEOR score between one candidate string and one or more reference strings.
    
    Args:
        reference_strs (str or List[str]): a single reference or a list of reference sentences.
        candidate_str (str): the candidate sentence to evaluate.
    
    Returns:
        float: the METEOR score.
    """
    # Normalize input to a list of reference strings
    if isinstance(reference_strs, str):
        references = [reference_strs]
    elif isinstance(reference_strs, list):
        references = reference_strs
    else:
        raise ValueError("reference_strs must be a string or a list of strings")
    
    # Compute and return METEOR
    return meteor_score(references, candidate_str)

# 示例用法
# references = [
#     'this is a great test ?',
#     'there are tests'
# ]
# candidate = 'this is a nice tests'
#
# bleu, bleu_1gram, bleu_2gram = calculate_bleu_scores(references, candidate)
# print(f'BLEU score: {bleu:.4f}')
# print(f'1-gram BLEU score: {bleu_1gram:.4f}')
# print(f'2-gram BLEU score: {bleu_2gram:.4f}')
