# from .vavanilla_resnet_cifar import vanilla_resnet20
from .vanilla_models.vanilla_resnet_cifar import vanilla_resnet20
from .vanilla_models.vanilla_resnet_imagenet import resnet18
from .vanilla_models.vanilla_mobilenet_imagenet import mobilenet_v2


from .quan_resnet_imagenet import resnet18_quan, resnet34_quan

from .quan_alexnet_imagenet import alexnet_quan

from .quan_mobilenet_imagenet import mobilenet_v2_quan

from .vanilla_models.vanilla_preactresnet_cifar import preactresnet18
from .quan_preactresnet_cifar import preactresnet18_quan

from .quan_resnet_cifar import resnet20_quan
# from .quan_resnet_cifar_brelu import resnet20_quan

from .quan_resnet_cifar import resnet32_quan
from .quan_wideresnet_cifar import wideresnet28_quan
from .quan_pyramidnet_cifar import pyramidnet110_quan
from .quan_resnet import resnet18_quan
from .quan_vgg_cifar import vgg19_bn_quan
# from .bin_resnet_cifar import resnet20_bin