import os

from torch import Tensor, nn
from sentence_transformers.models import Transformer, Pooling
from quaterion_models.types import TensorInterchange, CollateFnType
from quaterion_models.encoders import Encoder


class TweetsQAEncoder(Encoder):

    def __init__(self, transformer, pooling):
        super().__init__()
        self.transformer = transformer
        self.pooling = pooling
        self.encoder = nn.Sequential(self.transformer, self.pooling)

    @property
    def trainable(self) -> bool:
        # Defines if we want to train encoder itself, or head layer only
        return False

    @property
    def embedding_size(self) -> int:
        return self.transformer.get_word_embedding_dimension()

    def forward(self, batch: TensorInterchange) -> Tensor:
        return self.encoder(batch)["sentence_embedding"]

    def get_collate_fn(self) -> CollateFnType:
        # `collate_fn` is a function that converts input samples into Tensor(s) for use as encoder input.
        return self.transformer.tokenize

    @staticmethod
    def _transformer_path(path: str) -> str:
        # just an additional method to reduce amount of repeated code
        return os.path.join(path, "transformer")

    @staticmethod
    def _pooling_path(path: str) -> str:
        return os.path.join(path, "pooling")

    def save(self, output_path: str):
        # to provide correct saving of encoder layers we need to implement it manually
        transformer_path = self._transformer_path(output_path)
        os.makedirs(transformer_path, exist_ok=True)

        pooling_path = self._pooling_path(output_path)
        os.makedirs(pooling_path, exist_ok=True)

        self.transformer.save(transformer_path)
        self.pooling.save(pooling_path)

    @classmethod
    def load(cls, input_path: str) -> Encoder:
        transformer = Transformer.load(cls._transformer_path(input_path))
        pooling = Pooling.load(cls._pooling_path(input_path))
        return cls(transformer=transformer, pooling=pooling)


from typing import Union, Dict
from torch.optim import Adam
from quaterion import TrainableModel
from quaterion.loss import SimilarityLoss
from quaterion_models.heads import EncoderHead
from quaterion.train.cache import CacheConfig, CacheType
from quaterion.loss import MultipleNegativesRankingLoss
from quaterion_models.heads.skip_connection_head import SkipConnectionHead
from sentence_transformers import SentenceTransformer


class TweetsQAModel(TrainableModel):

    def __init__(self, model_name: str, *args, **kwargs):
        self.model_name = model_name
        super().__init__(*args, **kwargs)

    def configure_loss(self) -> SimilarityLoss:
        # `symmetric` means that we take into account correctness of both
        # the closest answer to a question and the closest question to an answer
        return MultipleNegativesRankingLoss(symmetric=True)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=10e-5)

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        base_model = SentenceTransformer(self.model_name)
        transformer: Transformer = base_model[0]
        pooling: Pooling = base_model[1]
        encoder = TweetsQAEncoder(transformer, pooling)
        return encoder

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        return SkipConnectionHead(input_embedding_size)

    def configure_caches(self):
        return CacheConfig(CacheType.AUTO, batch_size=256)