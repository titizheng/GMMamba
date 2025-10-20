
from models.classmodel import ClassMultiMILTransformer as ClassMMILT
from utils.utils import make_parse
from models.CSS import CSS_block ,Memory
from models.IMM import IMM_block





def create_model(args):

    model = IMM_block(in_dim = args.in_chans, n_classes=args.n_classes, dropout=args.drop_out, act='gelu', layer = args.mambamil_layer, mask_ratio = args.mask_ratio)
    basedmodel = model.cuda() 
    super_token_model = CSS_block(dim=512, n_classes = args.n_classes,n_iter=1) .cuda()
    classifymodel = ClassMMILT(args).cuda()
    memory = Memory()
    

    
    
    assert basedmodel is not None, "creating model failed. "
    print(f"basedmodel Total params: {sum(p.numel() for p in basedmodel.parameters()) / 1e6:.2f}M")
    print(f"classifymodel Total params: {sum(p.numel() for p in classifymodel.parameters()) / 1e6:.2f}M")
    return basedmodel,classifymodel,memory,super_token_model


if __name__ == "__mian__":
    
    args = make_parse()
    create_model(args)
