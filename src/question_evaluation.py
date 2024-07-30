from evaluation.question.length import get_question_length_histogram
from evaluation.question.latent_space import get_latent_space
from evaluation.question.cosine_similarity import get_cosine_similarity

if __name__ == "__main__":
    get_cosine_similarity(dir = "src/result/test5", filename = 'qna_context.json')