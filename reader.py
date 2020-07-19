import json
from overrides import overrides
import numpy as np

from allennlp.data import Token, Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, ListField, MetadataField, ArrayField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.util import START_SYMBOL, END_SYMBOL


@DatasetReader.register("edit-reader")
class EditReader(DatasetReader):
    def __init__(self, lazy: bool = False):
        super().__init__(lazy=lazy)
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    @staticmethod
    def _tokens_to_ids(tokens):
        ids = {}
        out = []
        for token in tokens:
            out.append(ids.setdefault(token.text, len(ids) + 1))
        return out

    @overrides
    def text_to_instance(self,
                         triple,
                         predicate,
                         draft,
                         revised=None,
                         action=None) -> Instance:
        triple_field = ListField([TextField(t, self.token_indexers) for t in triple])
        predicate_field = ListField([TextField(p, self.token_indexers) for p in predicate])
        draft.insert(0, Token(START_SYMBOL))
        draft.append(Token(END_SYMBOL))
        draft_field = TextField(draft, self.token_indexers)
        fields = {
            "triple_tokens": triple_field,
            "predicate_tokens": predicate_field,
            "draft_tokens": draft_field
        }
        meta_fields = {"draft": [w.text for w in draft[1:-1]], "triple": [t[-1].text for t in triple]}

        if revised is not None:
            meta_fields["revised"] = [w.text for w in revised]
            revised.insert(0, Token(START_SYMBOL))
            revised.append(Token(END_SYMBOL))

            action.insert(0, Token(START_SYMBOL))
            action.append(Token(END_SYMBOL))

            triple_revised_ids = self._tokens_to_ids([t[-1] for t in triple] + action)
            fields["triple_token_ids"] = ArrayField(np.array(triple_revised_ids[:len(triple)]))
            fields["action_token_ids"] = ArrayField(np.array(triple_revised_ids[len(triple):]))

            fields.update({"revised_tokens": TextField(revised, self.token_indexers),
                           "action_tokens": TextField(action, self.token_indexers)})
        else:
            fields["triple_token_ids"] = ArrayField(np.array(self._tokens_to_ids([t[-1] for t in triple])))

        fields["metadata"] = MetadataField(meta_fields)

        return Instance(fields)

    @overrides
    def _read(self, file_path):
        for ins in map(json.loads, open(file_path)):
            datum = {k: [Token(w) for x in ins[k] for w in x] for k in ("revised", "draft", "action")}
            datum["triple"] = [[Token(x[0]), Token(x[-1])] for x in ins["triple"]]
            datum["triple"].insert(0, [Token("@@EMPTY@@"), Token("@@EMPTY@@")])
            datum["predicate"] = [[Token(w) for w in x[1]] for x in ins["triple"]]
            datum["predicate"].insert(0, [Token("@@EMPTY@@")])

            yield self.text_to_instance(**datum)
