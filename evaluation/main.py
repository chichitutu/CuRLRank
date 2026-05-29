from transformers import HfArgumentParser
from runner import CuRLRankEvalArgs, CuRLRankEvalModelArgs, CuRLRankEvalRunner


def main():
    parser = HfArgumentParser((CuRLRankEvalArgs, CuRLRankEvalModelArgs))

    eval_args, model_args = parser.parse_args_into_dataclasses()
    eval_args: CuRLRankEvalArgs
    model_args: CuRLRankEvalModelArgs

    runner = CuRLRankEvalRunner(
        eval_args=eval_args,
        model_args=model_args
    )

    runner.run()


if __name__ == "__main__":
    main()
