from hipersim import MannTurbulenceField

#---Generate Turbulence Data---
def generate_mann_box(U_mean, desired_TI,
                      alphaepsilon, L, Gamma,
                      Nxyz=(16384, 32, 32), dxyz=(0.3632, 7.5, 7.5), seed=1):

    mtf = MannTurbulenceField.generate(alphaepsilon=alphaepsilon,
                                       L=L,
                                       Gamma=Gamma,
                                       Nxyz=Nxyz,
                                       dxyz=dxyz,
                                       seed=seed,
                                       HighFreqComp=0,
                                       double_xyz=(False, True, True))
    u, v, w = mtf.uvw

 
    # Compute and apply TI scaling
    #initial_TI = u.std(0).mean() / U_mean
    #scaling_factor = desired_TI / initial_TI
    #u *= scaling_factor
    #v *= scaling_factor
    #w *= scaling_factor
    return u, v, w, dxyz




def extract_params_from_file(filepath):
    df = pd.read_csv(filepath, header=None)
    L = float(df.iloc[7, 0])            # Row 8, column 1 (index 7, 0)
    alphaepsilon = float(df.iloc[9, 0]) # Row 10, column 1 (index 9, 0)
    Gamma = float(df.iloc[11, 0])       # Row 12, column 1 (index 11, 0)
    return alphaepsilon, L, Gamma


def extract_U_mean_from_filename(filename):
    try:
        parts = filename.split('_')
        for part in parts:
            if part.startswith('U'):
                return float(part[1:]) + 0.5
    except Exception:
        raise ValueError(f"Could not extract U_mean from filename {filename}")

def process_files_in_directory(directory, output_file="mann_TI_summary.txt"):
    results = []

    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    total_files = len(files)

    for i, filename in enumerate(files, 1):  # start counting at 1
        filepath = os.path.join(directory, filename)
        try:
            print(f"Processing file {i} of {total_files}: {filename}")

            alphaepsilon, L, Gamma = extract_params_from_file(filepath)
            U_mean = extract_U_mean_from_filename(filename)
            print(f"Parameters -> U_mean={U_mean}, αε={alphaepsilon}, L={L}, Γ={Gamma}")

            u, v, w, dxyz = generate_mann_box(U_mean=U_mean,
                                              desired_TI=0.1,
                                              alphaepsilon=alphaepsilon,
                                              L=L,
                                              Gamma=Gamma)

            TI_u = u.std(0).mean() / U_mean
            print(f"File: {filename} -> Realized TI (u): {TI_u:.4f}")

            results.append(f"{filename}\tU_mean={U_mean}\talphaepsilon={alphaepsilon}\tL={L}\tGamma={Gamma}\tTI={TI_u:.4f}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append(f"{filename}\tERROR: {e}")

    with open(output_file, 'w') as out_file:
        out_file.write("Filename\tU_mean\talphaepsilon\tL\tGamma\tTI\n")
        for line in results:
            out_file.write(line + "\n")

    print(f"\nSummary saved to {output_file}")

