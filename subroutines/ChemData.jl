# Functions for preparing the chemical data structure (read in from the HDF5 output from PySCF)


# The chemical data structure:
mutable struct ChemProperties
    mol_name::String
    basis::String
    geometry::String
    e_xhf::Float64
    
    e_fci::Float64
    fci_vec
    fci_str
    fci_addr
    e_nuc::Float64
    h1e
    h2e
    t2
    e_mo
    N_el::Int
    N::Int
    N_spt::Int
end
mutable struct ChemPropertiesUHF
    mol_name::String
    basis::String
    geometry::String
    e_xhf::Float64
    uhf_mos
    e_fci::Float64
    fci_vec
    alpha_occ_string
    beta_occ_string
    fci_str_a
    fci_str_b
    fci_addr_a
    fci_addr_b
    e_nuc::Float64
    h1e_a
    h1e_b
    h2e_aa
    h2e_bb
    h2e_ab
    t2
    e_mo
    N_el::Int
    N::Int
    N_spt::Int
end

# Read in the chemical data from the HDF5 directory:
function ReadIn(fname; uhf::Bool=false)
    
    fid = h5open(fname, "r")
    
    mol_name = read(fid, "mol_name")
    basis = read(fid, "basis")
    geometries = read(fid, "geometries")
    if uhf
        cdata_list = ChemPropertiesUHF[]
    else
        cdata_list = ChemProperties[]
    end
    for geometry in geometries
        
        grp = fid[geometry]
        #try e_xhf = read(grp, "e_rhf") catch
        e_xhf = read(grp, "uhf")
        #end
        
        if read(grp, "e_fci")=="N/A"
            e_fci = 0.0
            fci_vec = nothing
            fci_str = nothing
            fci_addr = nothing
        elseif uhf
            alpha_occ_string = read(grp, "alpha_occ_string")
            beta_occ_string = read(grp, "beta_occ_string")
            e_fci = read(grp, "e_fci")
            fci_vec = read(grp, "fci_obj")
            fci_str_a = read(grp, "fci_str_a")
            fci_str_b = read(grp, "fci_str_b")
            fci_addr_a = read(grp, "fci_addr_a")
            fci_addr_b = read(grp, "fci_addr_a")
            h1e_a = read(grp, "h1e_a")
            h1e_b = read(grp, "h1e_b")
            h2e_aa = read(grp, "h2e_aa")
            h2e_bb = read(grp, "h2e_bb")
            h2e_ab = read(grp, "h2e_ab")
            t2 = read(grp, "t2")
            e_mo = read(grp, "e_mo")
            N_el = read(grp, "nel")
            e_nuc = read(grp, "nuc")
            uhf_mos = read(grp, "uhf_mos")
        else
            e_fci = read(grp, "e_fci")
            fci_vec = read(grp, "fci_obj")
            fci_str = read(grp, "fci_str")
            fci_addr = read(grp, "fci_addr")
            h1e = read(grp, "h1e")
            h2e = read(grp, "h2e")
            e_mo = read(grp, "e_mo")
            N_el = read(grp, "nel")
            e_nuc = read(grp, "nuc")
        end
        
        N_spt = size(vcat(h1e_a,h1e_b), 1)
        N = 2*N_spt
        if uhf
            new_cdata = ChemPropertiesUHF(mol_name, basis, geometry, e_xhf, uhf_mos, e_fci, fci_vec, alpha_occ_string, beta_occ_string, fci_str_a,fci_str_b, fci_addr_a,fci_addr_b, e_nuc, h1e_a,h1e_b, h2e_aa,h2e_bb,h2e_ab, t2, e_mo, N_el, N, N_spt)
        else
            new_cdata = ChemProperties(mol_name, basis, geometry, e_xhf, e_fci, fci_vec, fci_str, fci_addr, e_nuc, h1e, h2e, t2, e_mo, N_el, N, N_spt)
        end
        push!(cdata_list, new_cdata)
        
    end

    close(fid)
    
    return cdata_list
    
end


# Reduce the dimensions to the specified active space:
function ActiveSpace(chemical_data, active)
    
    inert = setdiff(1:size(chemical_data.h1e,1), active)
    
    mol_name = chemical_data.mol_name
    basis = chemical_data.basis
    geometry = chemical_data.geometry
    e_xhf = chemical_data.uhf
    e_fci = chemical_data.e_fci
    e_nuc = chemical_data.e_nuc
    h1e = chemical_data.h1e[setdiff(1:end, inert),setdiff(1:end, inert)]
    h2e = chemical_data.h2e[setdiff(1:end, inert),setdiff(1:end, inert),setdiff(1:end, inert),setdiff(1:end, inert)]
    N_el = sum(active .<= floor(chemical_data.N_el/2)) + sum(active .<= ceil(chemical_data.N_el/2))
    N_spt = length(active)
    N = 2*length(active)
    
    #active_data = ChemProperties(mol_name, basis, geometry, e_xhf, e_fci, e_nuc, h1e, h2e, N_el, N, N_spt)
    
    #return active_data
    
end