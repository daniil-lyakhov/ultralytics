# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
import torch
from copy import deepcopy

import nncf
from nncf.torch.dynamic_graph.io_handling import FillerInputElement
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.graph.transformations.serialization import serialize_transformations
from nncf.torch.graph.transformations.serialization import load_transformations
from nncf.torch.graph.transformations.layout import TransformationLayout
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.nested_objects_traversal import objwalk
from nncf.common.factory import ModelTransformerFactory
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.graph.transformations.serialization import load_transformations
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTInsertionPoint
from nncf.torch.nncf_network import PTTargetPoint
from nncf.torch.nncf_network import TargetType
from nncf.common.factory import ModelTransformerFactory


NNCF_TRANSFORMATIONS_STATE_FILE_NAME = "nncf_transofmations_state.json"


def on_pretrain_routine_end(trainer):
    """Called before the pretraining routine starts."""

    if True:
        model = trainer.model
        loader = trainer.test_loader
        device = "cuda" if torch.cuda.is_available() else "cpu"
        def trainsform_fn(input_):
            #return 'https://ultralytics.com/images/bus.jpg'
            return torch.rand((1, 3,320,640)).to(device)
            return (objwalk(input_, lambda x: isinstance(x, torch.Tensor), lambda x: x.to(device).float()),)


        nncf_dataset = nncf.Dataset(loader, transform_func=trainsform_fn)
        # TODO actually quantize the model
        input_info = FillerInputInfo([FillerInputElement(shape=(1,3, 320, 640))])
        nncf_network = NNCFNetwork(deepcopy(model), input_info=input_info)
        class DummyOp(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._a = torch.nn.Parameter(torch.ones((1,)))

            def forward(self, x):
                return x

        breakpoint()

        layout = TransformationLayout()
        layout.register(PTSharedFnInsertionCommand([PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name='/nncf_model_input_0')], DummyOp(), "UNIQUE_OP_NAME"))
        transformer = ModelTransformerFactory.create(nncf_network)
        nncf_network = transformer.transform(layout)
        #trainer.model = nncf.quantize(deepcopy(model).to(device), calibration_dataset=nncf_dataset, subset_size=1)
        trainer.model = nncf_network
        return
    # resume from ckpt
    checkpoint = None
    breakpoint()
    with open(trainer.save_dir / NNCF_TRANSFORMATIONS_STATE_FILE_NAME, "r") as f:
        state_dict = json.load(f)
    transformations_layout = load_transformations(state_dict["NNCF_TRANSFORMATIONS_STATE"])
    input_info = FillerInputInfo.from_state(state_dict["NNCF_INPUT_INFO"])

    nncf_network = NNCFNetwork(deepcopy(trainer.model), input_info=input_info)
    model_transformer = ModelTransformerFactory.create(nncf_network)
    transformed_model = model_transformer.transform(transformations_layout)

    transformed_model.nncf.disable_dynamic_graph_building
    trainer.model = transformed_model


def on_model_save(trainer):
    """Called when the model is saved."""

    breakpoint()
    #nncf_transformations = model.nncf.get_applied_transformation_layout()
    nncf_transformations = TransformationLayout()

    callback_ckpt = dict()
    callback_ckpt["NNCF_TRANSFORMATIONS_STATE"] = serialize_transformations(nncf_transformations)

    input_info = FillerInputInfo([FillerInputElement(shape=(1, 3, 5, 5))])
    callback_ckpt["NNCF_INPUT_INFO"] = input_info.get_state()
    with open(trainer.save_dir / NNCF_TRANSFORMATIONS_STATE_FILE_NAME, "w") as f:
        json.dump(callback_ckpt, f)

callbacks = {
 "on_pretrain_routine_end": on_pretrain_routine_end,
 "on_model_save": on_model_save
}