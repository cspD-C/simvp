from .simvp import SimVP


def build_model(args):
    if args.model == "simvp":
        return SimVP(
            shape_in=tuple(args.in_shape),
            hid_s=args.hid_S,
            hid_t=args.hid_T,
            n_s=args.N_S,
            n_t=args.N_T,
            groups=args.groups,
        )
    raise ValueError(f"Unsupported model: {args.model}")
