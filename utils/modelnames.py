from models.item_pop import ItemPop
from models.user_pop import UserPop
from models.ncf import NCF
from models.vncf import VNCF
from models.e_ncf import ENCF
from models.e_vncf import EVNCF
from models.ce_ncf import CENCF
from models.ce_vncf import CEVNCF


models = {
    "NCF": NCF,
    "VNCF": VNCF,
    "E-NCF": ENCF,
    "E-VNCF": EVNCF,
    "CE-NCF": CENCF,
    "CE-VNCF": CEVNCF
}

explanable_models = {
    "ItemPop": ItemPop,
    "UserPop": UserPop,
    "E-NCF": ENCF,
    "E-VNCF": EVNCF,
    "CE-NCF": CENCF,
    "CE-VNCF": CEVNCF
}

critiquing_models = {
    "CE-NCF": CENCF,
    "CE-VNCF": CEVNCF
}
