import numpy as np

unknown = 1.999999
BOHR = 0.52917721092  # Angstroms

# fmt: off

# from Gerald Knizia's CtDftGrid, which is based on
#       http://en.wikipedia.org/wiki/Covalent_radius
# and
#       Beatriz Cordero, Veronica Gomez, Ana E. Platero-Prats, Marc Reves,
#       Jorge Echeverria, Eduard Cremades, Flavia Barragan and Santiago
#       Alvarez.  Covalent radii revisited. Dalton Trans., 2008, 2832-2838,
#       doi:10.1039/b801115j
COVALENT = 1/BOHR * np.array((unknown,  # Ghost atom
        0.31,                                     0.28,             # 1s
        1.28, 0.96, 0.84, 0.73, 0.71, 0.66, 0.57, 0.58,             # 2s2p
        1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,             # 3s3p
        2.03, 1.76,                                                 # 4s
        1.70, 1.60, 1.53, 1.39, 1.50, 1.42, 1.38, 1.24, 1.32, 1.22, # 3d
                    1.22, 1.20, 1.19, 1.20, 1.20, 1.16,             # 4p
        2.20, 1.95,                                                 # 5s
        1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, # 4d
                    1.42, 1.39, 1.39, 1.38, 1.39, 1.40,             # 5p
        2.44, 2.15,                                                 # 6s
        2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98,                   # La, Ce-Eu
        1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,             # Gd, Tb-Lu
              1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, # 5d
                    1.45, 1.46, 1.48, 1.40, 1.50, 1.50,             # 6p
        2.60, 2.21,                                                 # 7s
        2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69))

VDW = 1/BOHR * np.array((unknown,  # Ghost atom
    1.20,       #  1 H
    1.40,       #  2 He [1]
    1.82,       #  3 Li [1]
    1.53,       #  4 Be [5]
    1.92,       #  5 B  [5]
    1.70,       #  6 C  [1]
    1.55,       #  7 N  [1]
    1.52,       #  8 O  [1]
    1.47,       #  9 F  [1]
    1.54,       # 10 Ne [1]
    2.27,       # 11 Na [1]
    1.73,       # 12 Mg [1]
    1.84,       # 13 Al [5]
    2.10,       # 14 Si [1]
    1.80,       # 15 P  [1]
    1.80,       # 16 S  [1]
    1.75,       # 17 Cl [1]
    1.88,       # 18 Ar [1]
    2.75,       # 19 K  [1]
    2.31,       # 20 Ca [5]
    unknown,    # 21 Sc
    unknown,    # 22 Ti
    unknown,    # 23 V
    unknown,    # 24 Cr
    unknown,    # 25 Mn
    unknown,    # 26 Fe
    unknown,    # 27 Co
    1.63,       # 28 Ni [1]
    1.40,       # 29 Cu [1]
    1.39,       # 30 Zn [1]
    1.87,       # 31 Ga [1]
    2.11,       # 32 Ge [5]
    1.85,       # 33 As [1]
    1.90,       # 34 Se [1]
    1.85,       # 35 Br [1]
    2.02,       # 36 Kr [1]
    3.03,       # 37 Rb [5]
    2.49,       # 38 Sr [5]
    unknown,    # 39 Y
    unknown,    # 40 Zr
    unknown,    # 41 Nb
    unknown,    # 42 Mo
    unknown,    # 43 Tc
    unknown,    # 44 Ru
    unknown,    # 45 Rh
    1.63,       # 46 Pd [1]
    1.72,       # 47 Ag [1]
    1.58,       # 48 Cd [1]
    1.93,       # 49 In [1]
    2.17,       # 50 Sn [1]
    2.06,       # 51 Sb [5]
    2.06,       # 52 Te [1]
    1.98,       # 53 I  [1]
    2.16,       # 54 Xe [1]
    3.43,       # 55 Cs [5]
    2.49,       # 56 Ba [5]
    unknown,    # 57 La
    unknown,    # 58 Ce
    unknown,    # 59 Pr
    unknown,    # 60 Nd
    unknown,    # 61 Pm
    unknown,    # 62 Sm
    unknown,    # 63 Eu
    unknown,    # 64 Gd
    unknown,    # 65 Tb
    unknown,    # 66 Dy
    unknown,    # 67 Ho
    unknown,    # 68 Er
    unknown,    # 69 Tm
    unknown,    # 70 Yb
    unknown,    # 71 Lu
    unknown,    # 72 Hf
    unknown,    # 73 Ta
    unknown,    # 74 W
    unknown,    # 75 Re
    unknown,    # 76 Os
    unknown,    # 77 Ir
    1.75,       # 78 Pt [1]
    1.66,       # 79 Au [1]
    1.55,       # 80 Hg [1]
    1.96,       # 81 Tl [1]
    2.02,       # 82 Pb [1]
    2.07,       # 83 Bi [5]
    1.97,       # 84 Po [5]
    2.02,       # 85 At [5]
    2.20,       # 86 Rn [5]
    3.48,       # 87 Fr [5]
    2.83,       # 88 Ra [5]
    unknown,    # 89 Ac
    unknown,    # 90 Th
    unknown,    # 91 Pa
    1.86,       # 92 U [1]
    unknown,    # 93 Np
    unknown,    # 94 Pu
    unknown,    # 95 Am
    unknown,    # 96 Cm
    unknown,    # 97 Bk
    unknown,    # 98 Cf
    unknown,    # 99 Es
    unknown,    #100 Fm
    unknown,    #101 Md
    unknown,    #102 No
    unknown,    #103 Lr
))

sentinel = -1.0

# Recommended dipole polarizabilities \alpha_D (a.u.) indexed by Z (Z=1 to 120)
POLARIZABILITIES = np.array([
    sentinel,   # Z = 0
    4.50711,    # Z = 1 (H)
    1.38375,    # Z = 2 (He)
    164.1125,   # Z = 3 (Li)
    37.74,      # Z = 4 (Be)
    20.5,       # Z = 5 (B)
    11.3,       # Z = 6 (C)
    7.4,        # Z = 7 (N)
    5.3,        # Z = 8 (O)
    3.74,       # Z = 9 (F)
    2.6611,     # Z = 10 (Ne)
    162.7,      # Z = 11 (Na)
    71.2,       # Z = 12 (Mg)
    57.8,       # Z = 13 (Al)
    37.3,       # Z = 14 (Si)
    25.0,       # Z = 15 (P)
    19.4,       # Z = 16 (S)
    14.6,       # Z = 17 (Cl)
    11.083,     # Z = 18 (Ar)
    289.7,      # Z = 19 (K)
    160.8,      # Z = 20 (Ca)
    97.0,       # Z = 21 (Sc)
    100.0,      # Z = 22 (Ti)
    87.0,       # Z = 23 (V)
    83.0,       # Z = 24 (Cr)
    68.0,       # Z = 25 (Mn)
    62.0,       # Z = 26 (Fe)
    55.0,       # Z = 27 (Co)
    49.0,       # Z = 28 (Ni)
    46.5,       # Z = 29 (Cu)
    38.67,      # Z = 30 (Zn)
    50.0,       # Z = 31 (Ga)
    40.0,       # Z = 32 (Ge)
    30.0,       # Z = 33 (As)
    28.9,       # Z = 34 (Se)
    21.0,       # Z = 35 (Br)
    16.78,      # Z = 36 (Kr)
    319.8,      # Z = 37 (Rb)
    197.2,      # Z = 38 (Sr)
    162.0,      # Z = 39 (Y)
    112.0,      # Z = 40 (Zr)
    98.0,       # Z = 41 (Nb)
    87.0,       # Z = 42 (Mo)
    79.0,       # Z = 43 (Tc)
    72.0,       # Z = 44 (Ru)
    66.0,       # Z = 45 (Rh)
    26.14,      # Z = 46 (Pd)
    55.0,       # Z = 47 (Ag)
    46.0,       # Z = 48 (Cd)
    65.0,       # Z = 49 (In)
    53.0,       # Z = 50 (Sn)
    43.0,       # Z = 51 (Sb)
    38.0,       # Z = 52 (Te)
    32.9,       # Z = 53 (I)
    27.32,      # Z = 54 (Xe)
    400.9,      # Z = 55 (Cs)
    272.0,      # Z = 56 (Ba)
    215.0,      # Z = 57 (La)
    205.0,      # Z = 58 (Ce)
    216.0,      # Z = 59 (Pr)
    208.0,      # Z = 60 (Nd)
    200.0,      # Z = 61 (Pm)
    192.0,      # Z = 62 (Sm)
    184.0,      # Z = 63 (Eu)
    158.0,      # Z = 64 (Gd)
    170.0,      # Z = 65 (Tb)
    163.0,      # Z = 66 (Dy)
    156.0,      # Z = 67 (Ho)
    150.0,      # Z = 68 (Er)
    144.0,      # Z = 69 (Tm)
    139.0,      # Z = 70 (Yb)
    137.0,      # Z = 71 (Lu)
    103.0,      # Z = 72 (Hf)
    74.0,       # Z = 73 (Ta)
    68.0,       # Z = 74 (W)
    62.0,       # Z = 75 (Re)
    57.0,       # Z = 76 (Os)
    54.0,       # Z = 77 (Ir)
    48.0,       # Z = 78 (Pt)
    36.0,       # Z = 79 (Au)
    33.91,      # Z = 80 (Hg)
    50.0,       # Z = 81 (Tl)
    47.0,       # Z = 82 (Pb)
    48.0,       # Z = 83 (Bi)
    44.0,       # Z = 84 (Po)
    42.0,       # Z = 85 (At)
    35.0,       # Z = 86 (Rn)
    317.8,      # Z = 87 (Fr)
    246.0,      # Z = 88 (Ra)
    203.0,      # Z = 89 (Ac)
    217.0,      # Z = 90 (Th)
    154.0,      # Z = 91 (Pa)
    129.0,      # Z = 92 (U)
    151.0,      # Z = 93 (Np)
    132.0,      # Z = 94 (Pu)
    131.0,      # Z = 95 (Am)
    144.0,      # Z = 96 (Cm)
    125.0,      # Z = 97 (Bk)
    122.0,      # Z = 98 (Cf)
    118.0,      # Z = 99 (Es)
    113.0,      # Z = 100 (Fm)
    109.0,      # Z = 101 (Md)
    110.0,      # Z = 102 (No)
    320.0,      # Z = 103 (Lr)
    112.0,      # Z = 104 (Rf)
    42.0,       # Z = 105 (Db)
    40.0,       # Z = 106 (Sg)
    38.0,       # Z = 107 (Bh)
    36.0,       # Z = 108 (Hs)
    34.0,       # Z = 109 (Mt)
    32.0,       # Z = 110 (Ds)
    32.0,       # Z = 111 (Rg)
    28.0,       # Z = 112 (Cn)
    29.0,       # Z = 113 (Nh)
    31.0,       # Z = 114 (Fl)
    71.0,       # Z = 115 (Mc)
    sentinel,   # Z = 116 (Lv)
    76.0,       # Z = 117 (Ts)
    58.0,       # Z = 118 (Og)
    169.0,      # Z = 119 (Uue)
    159.0,      # Z = 120 (Ubn)
])

def calculate_NA(Z):
    """
    Calculates the standard chemical valence electron count for atomic number Z.
    Includes active d and f electrons, but buries them in the core once filled.
    """
    if Z < 0 or Z >= 121:
        return sentinel

    if Z == 0:
        return 0
    
    # Period 1 (H - He)
    if Z <= 2:
        return Z
    # Period 2 (Li - Ne)
    if Z <= 10:
        return Z - 2
    # Period 3 (Na - Ar)
    if Z <= 18:
        return Z - 10
    
    # Period 4
    if Z <= 30:
        return Z - 18  # K - Zn (4s, 3d are valence)
    if Z <= 36:
        return Z - 28  # Ga - Kr (3d is full and collapses to core)
    
    # Period 5
    if Z <= 48:
        return Z - 36  # Rb - Cd (5s, 4d are valence)
    if Z <= 54:
        return Z - 46  # In - Xe (4d is full and collapses to core)
    
    # Period 6
    if Z <= 56:
        return Z - 54  # Cs - Ba (6s are valence)
    if Z <= 71:
        return Z - 54  # La - Lu (Lanthanides: 6s, 5d, 4f are valence)
    if Z <= 80:
        return Z - 68  # Hf - Hg (4f is full and collapses, 6s, 5d are valence)
    if Z <= 86:
        return Z - 78  # Tl - Rn (4f, 5d are full and collapse)
    
    return sentinel

VALENCE_ELECTRONS = np.array([calculate_NA(Z) for Z in range(120 + 1)])

# fmt: on
