from evaluation.answer.cosine_similarity import get_answer_cosine_similarity_csv
from evaluation.answer.length import get_answer_length_histogram_csv
from evaluation.answer.latent_space import get_answer_latent_space_csv
from evaluation.question.cosine_similarity import get_question_cosine_similarity_csv
from evaluation.question.length import get_question_length_histogram_csv
from evaluation.question.latent_space import get_question_latent_space_csv

def main(dir: str = "src/result/test18"):
    get_answer_cosine_similarity_csv(dir=dir)
    get_answer_length_histogram_csv(dir=dir)
    get_answer_latent_space_csv(dir=dir)
    get_question_cosine_similarity_csv(dir=dir)
    get_question_length_histogram_csv(dir=dir)
    get_question_latent_space_csv(dir=dir)


if __name__ == "__main__":
    main("src/result/test23")
