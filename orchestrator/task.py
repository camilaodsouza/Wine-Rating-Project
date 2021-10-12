import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )

class CleanDataSet(DockerTask):


    in_csv = luigi.Parameter(default='/usr/share/data/raw/wine_dataset.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/interim/')
    #flag = luigi.Parameter('.SUCCESS_CleanDataset')

    @property
    def image(self):
        return f'code-challenge/clean-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        # TODO: implement correct command
        # Try to get the input path from self.requires() ;)
        return [
            'python', 'clean_dataset.py',
            '--in-csv', self.in_csv,
            '--out-dir', self.out_dir,
            #'--flag', self.flag
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class MakeDataset(DockerTask):

    in_csv = luigi.Parameter(default='/usr/share/data/interim/clean.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/partition/')
    #flag = luigi.Parameter('.SUCCESS_MakeDatasets')

    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    def requires(self):
        return CleanDataSet()

    @property
    def command(self):
        # TODO: implement correct command
        # Try to get the input path from self.requires() ;)
        return [
            'python', 'make_dataset.py',
            '--in-csv', self.in_csv,
            '--out-dir', self.out_dir,
            #'--flag', self.flag
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )

class TrainModel(DockerTask):

    train_csv = luigi.Parameter(default='/usr/share/data/partition/train.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/model/')
    #flag = luigi.Parameter('.SUCCESS_MakeDatasets')

    @property
    def image(self):
        return f'code-challenge/train_model:{VERSION}'

    def requires(self):
        return MakeDataset()

    @property
    def command(self):
        # TODO: implement correct command
        # Try to get the input path from self.requires() ;)
        return [
            'python', 'train_model.py',
            '--train-csv', self.train_csv,
            '--out-dir', self.out_dir,
            #'--flag', self.flag
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )

class EvaluateModel(DockerTask):

    in_model = luigi.Parameter(default='/usr/share/data/model/trained_model.sav')
    test_csv = luigi.Parameter(default='/usr/share/data/partition/test.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/output/')
    #flag = luigi.Parameter('.SUCCESS_MakeDatasets')

    @property
    def image(self):
        return f'code-challenge/evaluate_model:{VERSION}'

    def requires(self):
        return TrainModel()

    @property
    def command(self):
        # TODO: implement correct command
        # Try to get the input path from self.requires() ;)
        return [
            'python', 'evaluate_model.py',
            '--in-model', self.in_model,
            '--test-csv', self.test_csv,
            '--out-dir', self.out_dir,
            #'--flag', self.flag
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )
