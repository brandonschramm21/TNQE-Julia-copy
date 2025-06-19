# Python functions to run HF and FCI calculations, and save the output in HDF5 format

# Packages:
import pyscf
from pyscf import fci, ao2mo, lo, mp
from pyscf.tools import cubegen
from pyscf import gto, scf
from pyscf.scf import stability
import numpy as np
import h5py
import datetime
import os
import configparser


# Runs the PySCF calculations and saves the output to a HDF5 file:
def RunPySCF(config, gen_cubes=False, nosec=False):
    #pull all the configurations from config file "xyz.ini"
    mol_name = config['MOLECULE PROPERTIES']['mol_name']
    mol_spin = config['MOLECULE PROPERTIES'].getint('mol_spin', fallback=0)
    mol_charge = config['MOLECULE PROPERTIES'].getint('mol_charge', fallback=0)
    basis = config['CALCULATION PARAMETERS']['basis']
    run_rohf = config['CALCULATION PARAMETERS'].getboolean('run_rohf', fallback=False)
    run_uhf = config['CALCULATION PARAMETERS'].getboolean('run_uhf', fallback=False)
    init_hispin = config['CALCULATION PARAMETERS'].getboolean('init_hispin', fallback=False)
    init_prev = config['CALCULATION PARAMETERS'].getboolean('init_prev', fallback=False)
    run_fci = config['CALCULATION PARAMETERS'].getboolean('run_fci', fallback=False)
    loc_orbs = config['CALCULATION PARAMETERS'].getboolean('loc_orbs', fallback=False)
    active_space = config['CALCULATION PARAMETERS'].getboolean('active_space', fallback=False)
    active_norb = config['CALCULATION PARAMETERS'].getint('active_norb', fallback=0)
    active_nel = config['CALCULATION PARAMETERS'].getint('active_nel', fallback=0)
    xyz_folder = config['GEOMETRIES']['xyz_folder']
    xyz_files = config['GEOMETRIES']['xyz_files'].split(",")
    
    geometries = ["../configs/xyz_files/"+xyz_folder+"/"+xyz_file+".xyz" for xyz_file in xyz_files]
    
    wd = os.getcwd()+"/../datasets/pyscf_data/"
    wd_o = os.getcwd()+"/../datasets/orbs/"
    
    #define the filenames
    datestr = datetime.datetime.now()
    
    if nosec:
        filename = wd + mol_name + "_" + basis + "_" + datestr.strftime("%m%d%y%%%H%M") + ".hdf5"
    else:
        filename = wd + mol_name + "_" + basis + "_" + datestr.strftime("%m%d%y%%%H%M%S") + ".hdf5"
    
    #Perform SCF, MP2, and FCI calculations on each geometry
    #and save the results in HDF5 file
    with h5py.File(filename, 'w') as f:
        
        f.create_dataset("mol_name", data=mol_name)
        f.create_dataset("basis", data=basis)
        f.create_dataset("geometries", data=geometries)
    
        # Initialize these for later if you need to read in previous orbitals
        mo_init1 = None
        mocc_init1 = None
        
        for g, geometry in enumerate(geometries):
            print(geometry)
            #print(os.path.isfile(geometry))
            
            mol_obj = pyscf.gto.M(atom=geometry, basis=basis)
            mol_obj.basis = basis
            mol_obj.charge = mol_charge
            mol_obj.spin = mol_spin
            
            

            #RUNNING SELECTED SCF - more advanced options in RHF, implement to UHF as needed. --------------
            if run_rohf:
                print("Running ROHF")
                rhf_obj = pyscf.scf.ROHF(mol_obj)
                e_rhf = rhf_obj.kernel()
            elif run_uhf:
                mol_obj.symmetry = False
                #define SCF parameters
                uhf_obj1 = pyscf.scf.UHF(mol_obj).newton()
                uhf_obj1.conv_tol=1e-8
                uhf_obj1.max_cycle = 100
                e_uhf = uhf_obj1.kernel()

                # Perform stability analysis due to far out geometries
                mo_i, mo_e, stable_i, stable_e = pyscf.scf.stability.uhf_stability(uhf_obj1, return_status=True)
                

                while not stable_i:
                    print("Unstable solution found. Re-optimizing with stable(ish) orbitals...")
                    uhf_obj1.mo_coeff = mo_i
                    uhf_obj1.kernel()
                    mo_i, mo_e, stable_i, stable_e = pyscf.scf.stability.uhf_stability(uhf_obj1, return_status=True)
                else:
                    print("UHF solution is stable.")

                #obtain bitstring for the orbital occupancies for MPS construction later
                alpha_occ_indices = [i for i, occ in enumerate(uhf_obj1.mo_occ[0]) if occ > 0.5]
                beta_occ_indices = [i for i, occ in enumerate(uhf_obj1.mo_occ[1]) if occ > 0.5]
                print("index occupancies")
                print(alpha_occ_indices)
                print(beta_occ_indices)
            else:
                print("Running RHF")
                rhf_obj1 = pyscf.scf.RHF(mol_obj)
                rhf_obj1.DIIS = pyscf.scf.ADIIS
                rhf_obj1.conv_tol=1e-10
                
                if init_prev and (g > 0):
                    
                    e_rhf1 = rhf_obj1.kernel(c0=mo_init1)
                
                else: 

                    if init_hispin:

                        mol_hispin = pyscf.gto.M(atom=geometry, basis=basis)
                        mol_hispin.basis = basis
                        mol_hispin.charge = mol_charge
                        mol_hispin.spin = mol_hispin.nelectron

                        rhf_obj2 = pyscf.scf.RHF(mol_hispin)
                        rhf_obj2.DIIS = pyscf.scf.ADIIS
                        rhf_obj2.conv_tol=1e-10
                        rhf_obj2.init_guess = 'huckel'
                        e_rhf2 = rhf_obj2.kernel()

                        mo_init1 = rhf_obj2.mo_coeff
                        
                        e_rhf1 = rhf_obj1.kernel(c0=mo_init1)

                    else:

                        rhf_obj1.init_guess = 'huckel'
                        e_rhf1 = rhf_obj1.kernel()

                mocc_init = rhf_obj1.mo_occ
                mo_init = rhf_obj1.mo_coeff
                
                rhf_obj = pyscf.scf.RHF(mol_obj).newton()
                #rhf_obj2.DIIS = pyscf.scf.EDIIS
                rhf_obj.conv_tol=1e-11
                e_rhf = rhf_obj.kernel(mo_init,mocc_init)
                
                if init_prev:
                    mo_init1 = rhf_obj.mo_coeff
                    mocc_init1 = rhf_obj.mo_occ

            #RUNNING FCI --------------------------------------------------------

            if run_fci:
                if run_uhf:
                    print("Running UHF FCI")
                    #run the fci solver
                    cisolver = pyscf.fci.FCI(uhf_obj1)
                    e_fci, fci_obj = cisolver.kernel()
                    #get the determinant strings to make MPO later - one for each spin
                    norb_a = np.shape(uhf_obj1.mo_coeff[0])[0]
                    norb_b = np.shape(uhf_obj1.mo_coeff[1])[0]

                    fci_str_a_0 = fci.cistring.make_strings([*range(norb_a)], mol_obj.nelec[0])
                    fci_str_a = [bin(x) for x in fci_str_a_0]
                    fci_addr_a = [fci.cistring.str2addr(norb_a, mol_obj.nelec[0], x) for x in fci_str_a_0]
                    
                    fci_str_b_0 = fci.cistring.make_strings([*range(norb_b)], mol_obj.nelec[0])
                    fci_str_b = [bin(x) for x in fci_str_b_0]
                    fci_addr_b = [fci.cistring.str2addr(norb_b, mol_obj.nelec[1], x) for x in fci_str_b_0]
                    print(e_fci)
                else:
                    fci_obj = fci.FCI(rhf_obj)
                
                    fci_obj.conv_tol=1e-12
                    
                    e_fci, fci_obj = fci_obj.kernel()
                    if run_uhf:                    
                        norb = np.shape(uhf_obj1.mo_coeff)[0]
                    else:
                        norb = np.shape(rhf_obj1.mo_coeff)[0]
                    
                    fci_str0 = fci.cistring.make_strings([*range(norb)], mol_obj.nelec[0])
                    
                    fci_str = [bin(x) for x in fci_str0]
                    
                    fci_addr = [fci.cistring.str2addr(norb, mol_obj.nelec[0], x) for x in fci_str0]
                    
                    print(e_fci)
                
            else:
                
                e_fci, fci_obj, fci_str, fci_addr = "N/A", "N/A", "N/A", "N/A"
            assert e_fci != "N/A"    
            
            if loc_orbs:
                # C matrix stores the AO to localized orbital coefficients
                if run_uhf:
                    C = lo.pipek.PM(mol_obj).kernel(uhf_obj1.mo_coeff)
                else:
                    C = lo.pipek.PM(mol_obj).kernel(rhf_obj.mo_coeff)
                # Split-localization:
                """
                nocc = sum(rhf_obj.mo_occ>0.0)
                norb = len(rhf_obj.mo_occ)
                C = np.zeros((norb,norb))
                C[:,:nocc] = lo.pipek.PM(mol_obj).kernel(rhf_obj.mo_coeff[:,:nocc])
                C[:,nocc:] = lo.pipek.PM(mol_obj).kernel(rhf_obj.mo_coeff[:,nocc:])
                """
                
            #RUNNING MP2 --------------------------------------------------------
            if run_uhf:
                mp2_obj = mp.MP2(uhf_obj1).run()
            else:
                mp2_obj = mp.MP2(rhf_obj).run()
            t2 = mp2_obj.t2
            
            h1e = mol_obj.intor("int1e_kin") + mol_obj.intor("int1e_nuc")
            h2e = mol_obj.intor("int2e")
            
            # CONVERTING TO OTHER INTEGRAL BASIS (PROBABLY MO) -------------------
            if loc_orbs:
                scf_c = C
            else:
                if run_uhf:
                    # Get MO coefficients
                    C_alpha, C_beta = uhf_obj1.mo_coeff

                    # Transform one electron integrals to MO basis
                    h1e_alpha = C_alpha.T @ h1e @ C_alpha
                    h1e_beta  = C_beta.T  @ h1e @ C_beta

                    # Transform two electron integrals to MO basis - get three blocks aa, bb,ab
                    # For alpha-alpha spin block
                    h2e_aa = ao2mo.incore.full(h2e, C_alpha)

                    # For beta-beta spin block
                    h2e_bb = ao2mo.incore.full(h2e, C_beta)

                    # For alpha-beta spin block
                    h2e_ab = ao2mo.general(h2e, (C_alpha, C_alpha, C_beta, C_beta))  # (ij|ab)

                else:
                    scf_c = rhf_obj.mo_coeff
            
                    h1e = scf_c.T @ h1e @ scf_c
                    h2e = ao2mo.kernel(h2e, scf_c)
            if run_uhf:
                e_mo = uhf_obj1.mo_energy
            else:
                e_mo = rhf_obj.mo_energy
            


            #SAVING DATA IN HDF5 FILE -----------------------------------------------
            grp = f.create_group(geometry)
            if run_uhf:
                uhf_data = grp.create_dataset("uhf", data=e_uhf)
                uhf_mo_data = grp.create_dataset("uhf_mos", data=uhf_obj1.mo_coeff )
                a_occ_str = grp.create_dataset("alpha_occ_string", data=alpha_occ_indices)
                b_occ_str = grp.create_dataset("beta_occ_string", data=beta_occ_indices)
            else:
                rhf_data = grp.create_dataset("e_rhf", data=e_rhf)
            fci_data = grp.create_dataset("e_fci", data=e_fci)
            fci_objs = grp.create_dataset("fci_obj", data=fci_obj)
            if run_uhf:
                fci_strs_a = grp.create_dataset("fci_str_a", data=fci_str_a)
                fci_addr_as = grp.create_dataset("fci_addr_a", data=fci_addr_a)
                fci_strs_b = grp.create_dataset("fci_str_b", data=fci_str_b)
                fci_addr_bs = grp.create_dataset("fci_addr_b", data=fci_addr_b)
            else:
                fci_objs = grp.create_dataset("fci_obj", data=fci_obj)
                fci_strs = grp.create_dataset("fci_str", data=fci_str)
                fci_addr = grp.create_dataset("fci_addr", data=fci_addr)
            if run_uhf:
                h1e_alpha_data = grp.create_dataset("h1e_a", data=h1e_alpha)
                h1e_beta_data = grp.create_dataset("h1e_b", data=h1e_beta)
                h2e_aa_data = grp.create_dataset("h2e_aa", data=h2e_aa)
                h2e_bb_data = grp.create_dataset("h2e_bb", data=h2e_bb)
                h2e_ab_data = grp.create_dataset("h2e_ab", data=h2e_ab)
            else:
                h1e_data = grp.create_dataset("h1e", data=h1e)
                h2e_data = grp.create_dataset("h2e", data=h2e)
            t2_data = grp.create_dataset("t2", data=t2)
            e_mo = grp.create_dataset("e_mo", data=e_mo)
            nel_data = grp.create_dataset("nel", data=mol_obj.nelectron)
            nuc_data = grp.create_dataset("nuc", data=mol_obj.energy_nuc())
            
            if gen_cubes:
                
                dirpath_o = wd_o + mol_name + "_" + basis + "_" + datestr.strftime("%m%d%y%%%H%M%S")
                
                os.mkdir(dirpath_o)
                
                for i in range(scf_c.shape[1]):
                    fstring = dirpath_o + '/' + str(i+1).zfill(3) + '.cube'
                    cubegen.orbital(mol_obj, fstring, scf_c[:,i])