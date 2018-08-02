class Material:
    """
    Класс для хранения информации о материале
    """

    def __init__(self, name):
        self.name = name
        path = f'./user/mat_db/{self.name}.mtrl'
        d = {}
        with open(path) as f:
            for line in f:
                if not (line.startswith('//') or line == '\n' or line.startswith('#')):
                    key, val = line.split('=')
                    key, val = key.replace(' ', ''), val.replace(' ', '')
                    val = val.replace('\n', '')
                    d[key] = float(val)
            f.close()

        self.HEAT_CONDUCTIVITY = float(d['thermal_conductivity'])
        self.HEAT_CAPACITY = float(d['specific_heat_capacity'])
        self.HEAT_EXPANSION_COEFF = float(d['heatExpansionCoeff'])
        self.DENSITY = float(d['density'])
        self.PHONON_PORTION = float(d['phonon_portion'])

    def __repr__(self):
        rv = f'Material: {self.name}\nHEAT_CONDUCTIVITY: {self.HEAT_CONDUCTIVITY}\nHEAT_CAPACITY: {self.HEAT_CAPACITY}\nHEAT_EXPANSION_COEFF: {self.HEAT_EXPANSION_COEFF}\nDENSITY: {self.DENSITY}'
        return rv
