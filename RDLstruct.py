class Phenomenon:
    def __init__(self, name: str, ph_type: str, ph_from: str, ph_to: str, is_constraint: bool):
        self.name = name
        self.ph_type = ph_type
        self.ph_from = ph_from
        self.ph_to = ph_to
        self.is_constraint = is_constraint


class RDLStruct:
    def __init__(self, pattern: str, device: str = "",
                 input_data: list = None,
                 output_data: list = None,
                 req: str = "",
                 port: list = None):
        self.pattern = pattern
        self.device = device
        self.input_data = input_data or []
        self.output_data = output_data or []
        self.req = req
        self.port = port or []


def checkMatch(req_struct):
    IPlist = IP.getIP(req_struct.pattern)
    res_match = IP.get_matchIP(req_struct, IPlist)
    res_config = IP.get_configIP(req_struct, res_match[1])
    res_include = IP.get_includeIP(req_struct, res_config[1])
    res_custom = IP.get_customIP(req_struct, res_include[1])
    res = [res_match[0], res_config[0], res_include[0], res_custom[0]]
    return res


if __name__ == "__main__":
    my_struct = RDLStruct(
        pattern="GetData",
        device="DSS",
        output_data=["viewCode"],
        port=["0XC000"]
    )
    for key, value in vars(my_struct).items():
        print(f"{key}: {value}")
