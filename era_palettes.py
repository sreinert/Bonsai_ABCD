
# from colour import Color
from matplotlib.colors import LinearSegmentedColormap, to_hex

ERA_PALETTES = dict(
    Red1 = dict(colors=("#1E0502", "#2A0A04", "#370702", "#460902", "#5D1103", "#832F20", "#AC6356", "#CCA199", "#E9D9D7"), order=(5,7,1,9,3,6,8,2,4)   ),
    Red2 = dict(colors=("#4D5F63", "#370702", "#48402B", "#C89e82", "#914622", "#C1CFC4"), order=(1,3,5,6,2,4)   ),
    NineteenEightyNine = dict(colors=("#C34533", "#A55D38", "#CFA176", "#CACCC7", "#8CACC4", "#7AA1BD", "#5A88AE"), order=(1,5,3,6,4,7,2)   ),
    Showgirl1 = dict(colors=("#613A1B", "#642921", "#D07C40", "#6B7237", "#448363"), order=(5, 1, 3, 4, 2)   ),
    Showgirl2 = dict(colors=("#B74c2D", "#DC673E", "#CCB178", "#C1DCBF", "#7BB594", "#448363"), order=(1, 5, 3, 4, 6, 2)   ),
    SpeakNow1 = dict(colors=("#E2CFD8", "#C2A2B4", "#945791", "#7F407E", "#6B2D6D", "#421B4C", "#1C1120"), order=(2, 6, 1, 7, 4, 5, 3)   ),
    SpeakNow2 = dict(colors=("#F9DAC1", "#C27045", "#7E391D", "#29130F", "#050303"), order=(3, 1, 4, 5, 2)   ),
    TorturedPoet = dict(colors=("#EDECE8", "#E1DED7", "#B7B0A6", "#7F776D", "#443D35", "#342d26", "#2c251f"), order=(3,6,4,1,5,7,2)   ),
    Fearless = dict(colors=("#FAF2C8", "#EFDCA8", "#E0C082", "#B58F51", "#8A6737", "#664525", "#4b3220"), order=(7,3,5,1,6,4,2)   ),
    Evermore1 = dict(colors=("#D9BFAB", "#C3A38A", "#928171", "#514741", "#1D171B", "#51292A", "#451A11", "#71311D", "#AD5D3B"), order=(5,2,9,7,4,8,1,6,3)   ),
    Evermore2 = dict(colors=("#1D171B", "#451A11", "#71311D", "#AD5D3B", "#DD936E", "#D9BFAB"), order=(4,3,6,1,5,2)   ),
    Reputation = dict(colors=("#FEFEFE", "#BFBFBF", "#5D5D5D", "#2B2B2B", "#050505"), order=(5,3,1,2,4)   ),
    Lover1 = dict(colors=("#BF567F", "#FCB3C3", "#FEEFC6", "#815F51", "#4478ac"), order=(1, 5, 3, 4, 2)   ),
    Lover2 = dict(colors=("#BF567F", "#FAAFD3", "#D9A8E8", "#6CB4DC", "#4478ac"), order=(1, 5, 3, 4, 2)   ),
    TaylorSwift = dict(colors=("#142b1A", "#486833", "#689739", "#30C97E", "#29ADDE", "#2297B8", "#23677E"), order=(3,7,5,1,4,2,6)   ),
    Midnight1 = dict(colors=("#02091c", "#1B2541", "#354167", "#50658a", "#64829d", "#829eb1", "#a9b3c0"), order=(7,2,6,4,1,5,3)   ),
    Midnight2 = dict(colors=("#411D30", "#5F2732", "#9A383C", "#BC7E6A", "#CBCBCD", "#7191A9"), order=(3,5,1,6,4,2)   ),

  )

EXPORT_FORMATS = {"HEX", "DEC", "REL", "XML", "IPE"}

def hex2rgb(hex_color):
    """Convert a hex color to RGB format."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb2hex(rgb_color):
    """Convert an RGB color to hex format."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)


def era_brew(name, n=None, brew_type="discrete"):

    palette = ERA_PALETTES.get(name)

    if not brew_type or type(name) in (int, float):
        raise Exception(f"Palette {name} does not exist.")

    if not n:
        n = len(palette["colors"])
        print(f"Palette '{name}' has '{n}' discrete colors")


    if brew_type not in {"discrete", "continuous"}:
        brew_type = "discrete"

    if brew_type == "discrete" and n > len(palette["colors"]):
        print(f"Number of requested colors ('{n}') greater than number of colors '{name}' can offer. \n Setting brew_type to 'continuous' instead.")
        brew_type = "continuous"

    out = list()
    if brew_type == "continuous":

        color_map = LinearSegmentedColormap.from_list(name, palette["colors"], N=n)
        for i in range(n):
            out.append(to_hex(color_map(i)))


    elif brew_type == "discrete":

        rounds = n // len(palette["colors"])
        delta = n % len(palette["colors"])
        for _ in range(rounds):
            for i in range(len(palette["colors"])):
                idx = palette["order"][i] - 1
                color = palette["colors"][idx]
                out.append(color)

        for i in range(delta):
            idx = palette["order"][i] - 1
            color = palette["colors"][idx]
            out.append(color)

    return out

def export(name, format="hex"):

    format = format.upper()

    palette = ERA_PALETTES.get(name)
    colors = [hex2rgb(c) for c in palette["colors"]]

    export = dict()
    if palette and format in EXPORT_FORMATS:
        if format == "HEX":
            export = {
                "name": name,
                "colors": [rgb2hex(c) for c in colors]
            }

        elif format == "DEC":
            export = {
                "name": name,
                "colors": colors
            }

        elif format == "REL":
            export = {
                "name": name,
                "colors": colors
            }

        elif format in {"XML", "IPE"}:
            color_values = [
                tuple([round(v, 3) for v in c])
                for c in colors
            ]
            color_strings = [" ".join(str(v) for v in c) for c in color_values]
            export = {
                "name": name,
                "colors": color_values,
                "tags": [
                    f"<color name=\"{name}-{i}\" value=\"{v}\" />"
                    for i, v in enumerate(color_strings, start=1)
                ]
            }

        return export


