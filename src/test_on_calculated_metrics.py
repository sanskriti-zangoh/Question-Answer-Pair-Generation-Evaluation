from evaluation.answer.length import get_answer_length_histogram
from evaluation.question.length import get_question_length_histogram
from evaluation.question.cosine_similarity import get_cosine_similarity
from evaluation.question.latent_space import get_latent_space

dir = "src/result/test10"

get_latent_space(dir = dir, filename = 'qna_context_embeddings.csv')
get_cosine_similarity(dir = dir, filename = 'qna_context_embeddings.csv')
get_question_length_histogram(dir = dir, filename = 'qna_context_embeddings.csv')
get_answer_length_histogram(dir = dir, filename = 'qna_context_embeddings.csv')