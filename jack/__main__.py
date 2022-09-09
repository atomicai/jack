import functools

# fmt: off
import logging
import warnings
from pathlib import Path
from typing import List

import typer
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.utils import grouper
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import (calc_chunksize, initialize_device_settings,
                        set_all_seeds)

from jack.logging.module import logger as ai_logger
from jack.tooling import pipe

warnings.filterwarnings('ignore')
app = typer.Typer()


@app.command()
def train(model_name_or_path: str = "cointegrated/rubert-tiny", max_seq_len: int = 192, ):
    ##########################
    ########## Settings
    ##########################

    def _train(model, optimizer, data_silo, epochs, n_gpu, lr_schedule, evaluate_every, tracker, device):
        trainer = Trainer(
                    prefix="",
                    model=model,
                    optimizer=optimizer,
                    data_silo=data_silo,
                    epochs=n_epochs,
                    n_gpu=n_gpu,
                    lr_schedule=lr_schedule,
                    log_loss_every=1,
                    evaluate_every=evaluate_every,
                    tracker=tracker,
                    device=device)
        model = trainer.train()

        return model
    
    def _processing(pretrained_model_name_or_path, do_lower_case, label_list: List[str], metric: str, batch_size: int = 4):
        tokenizer = Tokenizer.load(pretrained_model_name_or_path, do_lower_case=do_lower_case, use_fast=True)

        processor = TextClassificationProcessor(
            tokenizer=tokenizer,
            max_seq_len=192,
            data_dir=Path.home() / "Dataset" / "rc",
            train_filename="x_july.csv",
            dev_filename="y_july.csv",
            test_filename="y_july.csv",
            label_list=label_list,
            metric=metric,
            dev_split=0.0,
            delimiter=",",
            dev_stratification=dev_stratification,
            text_column_name="text",
            label_column_name="label"
        )
        
        data_silo = DataSilo(
            processor=processor,
            max_processes=1,
            batch_size=batch_size
        )

        return processor, data_silo
    
    def _modeling(pretrained_model_name_or_path, loss_fn: str, device):
        language_model = LanguageModel.load(pretrained_model_name_or_path)
        # b) and a prediction head on top that is suited for our task => Text classification
        loss_fn = "crossentropy"
        prediction_head = TextClassificationHead(
            class_weights=data_silo.calculate_class_weights(task_name="text_classification"),
            num_labels=len(label_list),
            loss_fn=loss_fn
        )

        model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.1,
            lm_output_types=["per_sequence"],
            device=device)
        
        return model
    
    def _optimizing(model, device, use_amp, lr, n_epochs):
        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=3e-5,
            device=device,
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            use_amp=use_amp
        )
        return model, optimizer, lr_schedule
    

    set_all_seeds(seed=2077)
    n_epochs = 5
    batch_size = 4
    evaluate_every = 150
    data_dir=Path.home() / "Dataset" / "rc"
    train_filename="x_july.csv",
    dev_filename="y_july.csv",
    test_filename="y_july.csv",
    lang_model = "distilbert-base-multilingual-cased"
    do_lower_case = False
    dev_split = 0.0
    dev_stratification = True
    max_processes = 1    # 128 is default
    max_chunksize = 512
    save_dir=Path.home() / "Weights" / "output" / "mbert-distilled-cased_rc"
    # or a local path:
    # lang_model = Path("../saved_models/farm-bert-base-cased")
    use_amp = None

    device, n_gpu = initialize_device_settings(use_cuda=True, use_amp=use_amp)

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

    # for i, experiment in enumerate(range(2)):
    #     pass


    metric = "f1_micro"
    loss_fn = "crossentropy"
    learning_rate = 3e-5
    ######################
    ########## Processing
    ######################

    tokenizer, processor, data_silo = pipe._processing(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case,
        label_list=label_list, metric=metric,
        data_dir=data_dir,
        train_filename="x_july.csv",
        dev_filename="y_july.csv",
        test_filename="y_july.csv",
        dev_stratification=dev_stratification)

    # processor, data_silo = _processing(pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case, label_list=label_list, metric=metric, batch_size=batch_size)

    ######################
    ########## Modeling
    ######################

    model = pipe._modeling(pretrained_model_name_or_path=lang_model, data_silo=data_silo, loss_fn=loss_fn, label_list=label_list, device=device)

    ######################
    ########## Optimizing and Logging
    ######################

    # Can we use here simply `data_silo.batch_size` ?

    model, optimizer, lr_scheduler = pipe._optimizing(model=model, device=device, n_batches=len(data_silo.loaders["train"]), n_epochs=n_epochs, lr=learning_rate)
    
    # _optimizing(model=model, device=device, use_amp=use_amp, lr=learning_rate, n_epochs=n_epochs)

    project_name = "RC"
    experiment_name = "rc_july"

    tracker = pipe._logging(prefix=f"{loss_fn} ¬ ",project_name=project_name, experiment_name=experiment_name)

    model = _train(model=model, optimizer=optimizer, data_silo=data_silo, epochs=n_epochs, n_gpu=n_gpu, lr_schedule=lr_scheduler, evaluate_every=100, tracker=tracker, device=device)

    tracker.end_run()

    model.save(save_dir)
    processor.save(save_dir)
    # tokenizer.save_pretrained(save_dir)

    # # 5. Create an optimizer
    # model, optimizer, lr_schedule = initialize_optimizer(
    #     model=model,
    #     learning_rate=3e-5,
    #     device=device,
    #     n_batches=len(data_silo.loaders["train"]),
    #     n_epochs=n_epochs,
    #     use_amp=use_amp)

    # ########## and logger
    # API_KEY = "9b7524ccc0cc7f67444fa6d0662c993fba1dde33"
    # project_name = "RC"
    # experiment_name = "rc_july"
    # ml_logger = ai_logger.WANDBLogger.init_experiment(
    #     project_name=project_name,
    #     experiment_name=experiment_name,
    #     prefix=f"{loss_fn} # ",
    #     api=API_KEY,
    #     sync_step=False,
    # )

    # ######################
    # ########## Training
    # ######################
    
    # # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    # trainer = Trainer(
    #     prefix="",
    #     model=model,
    #     optimizer=optimizer,
    #     data_silo=data_silo,
    #     epochs=n_epochs,
    #     n_gpu=n_gpu,
    #     lr_schedule=lr_schedule,
    #     log_loss_every=1,
    #     evaluate_every=evaluate_every,
    #     tracker=ml_logger,
    #     device=device)
    
    # # 7. Let it grow
    # trainer.train()

    # ml_logger.end_run()




@app.command()
def evaluate(name: str, formal: bool = False):
    if formal:
        print(f"Goodbye Ms. {name}. Have a good day.")
    else:
        print(f"Bye {name}!")


if __name__ == "__main__":
    app()
