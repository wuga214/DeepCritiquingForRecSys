from models.cncf import CNCF
from models.cvncf import CVNCF
from models.incf import INCF
from models.ivncf import IVNCF
from models.ncf import NCF
from models.vncf import VNCF
from models.user_pop import UserPop
from models.item_pop import ItemPop

models = {
    "NCF": NCF,
    "INCF": INCF,
    "CNCF": CNCF,
    "VNCF": VNCF,
    "IVNCF": IVNCF,
    "CVNCF": CVNCF
}

explanable_models = {
    "INCF": INCF,
    "CNCF": CNCF,
    "IVNCF": IVNCF,
    "CVNCF": CVNCF,
    "UserPop": UserPop,
    "ItemPop": ItemPop
}