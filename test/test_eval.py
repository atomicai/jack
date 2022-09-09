from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.eval import Evaluator
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.tokenization import Tokenizer
from farm.utils import initialize_device_settings
from jack.tooling import pipe


def main():
    ##########################
    ########## Settings
    ##########################
    device, n_gpu = initialize_device_settings(use_cuda=True)
    # lang_model = "distilbert-base-multilingual-cased"
    lang_model = Path.home() / "Weights" / "output" / "mbert-distilled-cased_rc"
    do_lower_case = False
    batch_size = 100
    dev_stratification = True
    data_dir = Path.home() / "Dataset" / "rc"
    evaluation_filename = "x_july.csv"
    label_list = [
        "Вопросы по порядку начисления процентов/размере процентной ставки по накопительному счету или счету \"Активный возраст\" (почему такая ставка/почему уменьшается ставка/как начисляются проценты/как начисляются проценты при досрочном закрытии вклада/не согласен с начисленными процентами, но проценты начислены корректно и пояснен порядок начисления %)",
        "Подбор вклада/условия вклада, который планирует открыть (какие вклады есть в СБ/какой вклад выгодней/размер ставки/какой вклад подойдет для совершения определенной операции/условия вклада по поступившему предложению от банка/какие приходные или расходные операции доступны/ограничения по сумме открытия или пополнения)",
        "Способы или сроки закрытия вклада/можно ли закрыть вклад досрочно (в том числе, как закрыть в СБОЛ/МП/как закрыть валютный вклад без конвертации)",
        "Условия/порядок проведений операций по действующему вкладу (срок вклада/сколько средств должно храниться на вкладе, чтобы он не был закрыт/какие доступны операции/где можно снять средства со вклада/какие документы нужны для проведения операций по вкладу)",
        "Порядок начисления процентов по вкладам, кроме Накопительного счета и Активный возраст (когда или как начисляются проценты/периодичность капитализации/как начисляются проценты при досрочном закрытии вклада/какой процент будет при дополнительном взносе/не согласен с начисленными процентами, но проценты начислены корректно и пояснен порядок начисления)",
        "Порядок пролонгации вклада (нужно ли обращаться в ВСП для продления вклада/продлевается ли вклад автоматически/можно ли досрочно продлить вклад/можно ли не продлевать вклад/по какой ставке произойдет пролонгация вклада/меняются ли реквизиты при пролонгации/на какую сумму пролонгируется вклад, с учетом или без учета уже начисленных процентов/в какое время произойдет пролонгация)",
        "Порядок или способы открытия вклада/счета (как или где открыть/кто может открыть/какие документы нужны для открытия/пояснения по интерфейсу СБОЛ или МП при открытии)",
        "Как/когда возможно получить проценты или закрыть вклад, чтобы не потерять начисленные проценты (в т.ч. когда дата приходится на выходной/праздничный день)",
        "UNKNOWN",
    ]
    metric = "f1_macro"

    # 1. Create a Processor alongside the Ignitor

    tokenizer, processor, data_silo = pipe._processing(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case,
        label_list=label_list,
        metric=metric,
        data_dir=data_dir,
        train_filename=None,
        dev_filename=None,
        test_filename=evaluation_filename,
        dev_stratification=dev_stratification,
    )

    # # 1.Create a tokenizer
    # tokenizer = Tokenizer.load(pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case)

    # # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # # Here we load GermEval 2017 Data automaticaly if it is not available.

    # processor = TextClassificationProcessor(
    #     tokenizer=tokenizer,
    #     max_seq_len=192,
    #     label_list=label_list,
    #     metric=metric,
    #     train_filename=None,
    #     dev_filename=None,
    #     dev_split=0,
    #     test_filename=evaluation_filename,
    #     data_dir=data_dir,
    # )

    # 2. Create Evaluator object
    evaluator = Evaluator(data_loader=data_silo.get_data_loader("test"), tasks=data_silo.processor.tasks, device=device)

    # 3. Create a model with Projection Head

    model = pipe._modeling(
        pretrained_model_name_or_path=lang_model,
        data_silo=data_silo,
        loss_fn="crossentropy",
        label_list=label_list,
        device=device,
    )

    # 4. Let's connect model's head (e.g. `TextClassificationHead` )
    model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

    #
    print("something...")


if __name__ == "__main__":
    main()
