from evaluation.answer.cosine_similarity import get_answer_cosine_similarity_csv
from evaluation.answer.length import get_answer_length_histogram_csv
from evaluation.answer.latent_space import get_answer_latent_space_csv
from evaluation.question.cosine_similarity import get_question_cosine_similarity_csv
from evaluation.question.length import get_question_length_histogram_csv
from evaluation.question.latent_space import get_question_latent_space_csv
from logging import Logger

def main(logger: Logger, dir: str = "src/result/test18", filename: str = 'qna_overall.csv'):
    get_answer_cosine_similarity_csv(dir=dir, filename=filename, logger=logger)
    get_answer_length_histogram_csv(dir=dir, filename=filename, logger=logger)
    get_answer_latent_space_csv(dir=dir, filename=filename, logger=logger)
    get_question_cosine_similarity_csv(dir=dir, filename=filename, logger=logger)
    get_question_length_histogram_csv(dir=dir, filename=filename, logger=logger)
    get_question_latent_space_csv(dir=dir, filename=filename, logger=logger)
    logger.info("STATUS: metrics calculation done")


if __name__ == "__main__":
    main("src/result/test20")
