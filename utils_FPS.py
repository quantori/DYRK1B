import numpy as np
import pandas as pd
from rdkit.Chem import AllChem as Chem
import oddt
from oddt.shape import electroshape

from rdkit.Chem import ChemicalFeatures

SD_FP_LENGTH   = 322
ODDT_FP_LENGTH = 15

fdefName   = 'SDSMIRKS.fdef'
factory    = ChemicalFeatures.BuildFeatureFactory(fdefName)
dict_fdefs = factory.GetFeatureDefs()

#%%
def get_SD_fps( df_in ):
    """
    Generate fingerpritns via SMIRKS patterns:
    """
    global dict_fdefs

    if not dict_fdefs:
        factory = ChemicalFeatures.BuildFeatureFactory( fdefName )
        dict_fdefs = factory.GetFeatureDefs()

    fp4 = np.zeros( SD_FP_LENGTH )

    str_fp4 = [ 'SDFP' + str(i) for i in range(len(fp4)) ]
    cmb = ['Structure']
    for i in str_fp4: cmb.append(i)
    df2grow = pd.DataFrame( columns=cmb, index=df_in.index )

    print("Calculating SD fingerprints...")
    for idx_smile, smile in zip(np.arange(0, len(df_in.Structure.values)), df_in.Structure.values):  
        # print("   Calculating fps for smi =", smile)
        mol = Chem.MolFromSmiles( smile )
        kvf_sorted = sorted( [(int(k.split('.')[0])-1, v, len(mol.GetSubstructMatches(Chem.MolFromSmarts(v)))) for k,v in dict_fdefs.items()] )
        fp4 = np.zeros( SD_FP_LENGTH )
        for idx_feat, _, feat_val in kvf_sorted: fp4[ idx_feat ] = feat_val

        new_entry = [smile]
        for i in fp4: new_entry.append(i)
        df2grow.iloc[idx_smile] = new_entry

    print("SD fps were successfully generated")
    df_ret = df_in.merge( df2grow, how="inner", on=['Structure'] )

    return df_ret

def get_FPSD_from_mol( mol ):
    kvf_sorted = sorted( [(int(k.split('.')[0])-1, v, len(mol.GetSubstructMatches(Chem.MolFromSmarts(v)))) for k,v in dict_fdefs.items()] )
    
    fps_SD = np.zeros( SD_FP_LENGTH )
    for idx_feat, _, feat_val in kvf_sorted: 
        fps_SD[ idx_feat ] = feat_val
        
    return fps_SD


def get_RDkit_fps_by_name(smiles, fp_name='maccs', nbits=1024, longbits=16384, use3D=True):
    import numpy as np
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import MACCSkeys, AllChem
    from rdkit.Avalon import pyAvalonTools as fpAvalon
    # from rdkit.Chem.AtomPairs import Pairs, Torsions
    from rdkit.Chem import rdMolDescriptors    

    m = Chem.MolFromSmiles(smiles)
    if use3D == True :
        #print("   Converting to 3D, optimizing geometry...")
        m2 = Chem.AddHs(m)
        AllChem.EmbedMolecule(m2)
        # AllChem.UFFOptimizeMolecule(m2)
        AllChem.MMFFOptimizeMolecule(m2)
        m = m2
    
    # dictionary
    fpdict = {}
    fpdict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
    fpdict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
    fpdict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
    fpdict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
    #fpdict['ecfc0'] = lambda m: AllChem.GetMorganFingerprint(m, 0)
    #fpdict['ecfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1)
    #fpdict['ecfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2)
    #fpdict['ecfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3)
    fpdict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
    fpdict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
    fpdict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
    #fpdict['fcfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1, useFeatures=True)
    #fpdict['fcfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2, useFeatures=True)
    #fpdict['fcfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3, useFeatures=True)
    fpdict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
    fpdict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
    fpdict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=longbits)
    fpdict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=longbits)
    fpdict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
    #fpdict['ap'] = lambda m: Pairs.GetAtomPairFingerprint(m)
    #fpdict['tt'] = lambda m: Torsions.GetTopologicalTorsionFingerprintAsIntVect(m)
    fpdict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
    fpdict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
    fpdict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
    fpdict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, longbits)
    fpdict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
    fpdict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
    fpdict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)   
    
    if( not fp_name in fpdict.keys() ) :
         raise ValueError( fp_name + "<-- is not in the registered list of RDKit algos: \n" + "\n".join( sorted(fpdict.keys()) ) )
    else :         
        fp = fpdict[fp_name](m)
        fp_arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, fp_arr )
        
    return fp_arr
    
def get_RDkit_fps_by_list(smiles, fp_list=['maccs'], fp_all=False, nbits=1024, longbits=16384, use3D=True):
    fp_list_ALL = [
        'ecfp0',
        'ecfp2',
        'ecfp4',
        'ecfp6',
        #'ecfc0',
        #'ecfc2',
        #'ecfc4',
        #'ecfc6',
        'fcfp2',
        'fcfp4',
        'fcfp6',
        #'fcfc2',
        #'fcfc4',
        #'fcfc6',
        'lecfp4',
        'lecfp6',
        'lfcfp4',
        'lfcfp6',
        'maccs',
        #'ap',
        #'tt',
        'hashap',
        'hashtt',
        'avalon',
        'laval',
        'rdk5',
        'rdk6',
        'rdk7'
    ]    
    
    if fp_all == True :
        fp_list = fp_list_ALL
    
    fp_arr = np.array([])        
        
    for fp_name in fp_list :

        if( not fp_name in fp_list_ALL ) :
            raise ValueError( fp_name + "<-- is not in the registered list of RDKit algos: \n" + "\n".join( sorted(fp_list_ALL) ) )
        else :
            fp_arr = np.hstack( (fp_arr, get_RDkit_fps_by_name(smiles, fp_name, nbits, longbits, use3D)) )
    
    return fp_arr
    
def get_RDkit_fps( df_in, fp_list=['maccs'], fp_all=True, nbits=1024, longbits=16384, use3D=True ) :
    FP_LENGTH_RDKIT_ALL = 95399
    fp4 = np.zeros( FP_LENGTH_RDKIT_ALL )
    
    print("Preparing data structures...")
    str_fp4 = [ 'RDkitFP' + str(i) for i in range(len(fp4)) ]
    cmb = ['Structure']
    for i in str_fp4: cmb.append(i)
    FP_names = cmb[1:]
    df2grow = pd.DataFrame( columns=cmb, index=np.arange(0, len(df_in['Structure'].values)) )

    print('Calculating requested RDkit fingerprints...')
    for idx_smile, smile in enumerate(df_in['Structure']) :
        print("Processing entry # " + str(idx_smile) + " -->" + smile)    
        FP_vals = get_RDkit_fps_by_list(smile, fp_all=True, use3D=True, nbits=1024, longbits=16384)
        df2grow.loc[idx_smile]['Structure'] = smile
        df2grow.loc[idx_smile][FP_names] = FP_vals
        # df2grow.append(df_2add, ignore_index=True)
    
    df_ret = pd.merge(df_in, df2grow, on=['Structure'])
        
    print("RDKit fingerprints were successfully generated\n")
    return df_ret

def fp_qsar_input( df_in, FP_type = "SD" ):
    '''
    Currently supports ONLY SD fingerpritns
    '''
    df_QSAR_fps = get_SD_fps( df_in[["Structure","ID","Potency"]] ) # if want StarDrop fps
    
    return df_QSAR_fps

def get_fp_oddt_from_mol_conf( m, calccharges = True ):
    '''Calculate ODDT fingerprint from an RDkit conformer'''
    # m = Chem.AddHs( m ) # Necessary step, BUT ONLY IF input mol does NOT have Hs added (otherwise RDkit fails to add them twice)
    str_conf = Chem.MolToMolBlock( m )
    oddt_conf = oddt.toolkit.readstring('sdf', str_conf)
    # print("str_conf =", str_conf)
    if calccharges:
        oddt_conf.calccharges() # will generate TOTALLY DIFFERENT fps!!! (those MIGHT BE BETTER FPS!)
        
    fp_oddt = electroshape(oddt_conf)
    
    return fp_oddt

def get_mol_from_SDF( fname_in_SDF, first_only = True ):
    '''get RDkit mol obj from 3D SDF / check for 3D'''
    suppl = Chem.SDMolSupplier( fname_in_SDF ) # 'cs_4screen.sdf')
    query_mols = [] # mol container
    
    for idx_m, m in enumerate( suppl ):
        # check for 3D:
        try:
            is3D = m.GetConformer().Is3D()
        except:
            print("Unable to check m.GetConformer().Is3D() for fname_in_SDF =", fname_in_SDF)
            m = None
            if first_only:
                return m
            else:
                query_mols.append( m )
                continue
        
        m_name = m.GetProp("_Name")
        # print( "compound", idx_m, "from", os.path.basename(fname_in_SDF), "is 3D :", is3D ) # , Chem.MolToMolBlock(m).split("\n")[1])
        if not is3D:
            print("  ERROR: conformer", m_name, "is NOT 3D")
            # sys.exit(1)
        else:
            query_mols.append( m )
            
        if first_only:
            return query_mols[0]
        
    return query_mols

def df_column_switch(df, name_col1, name_col2):
    list_c = list(df.columns)
    idx_1, idx_2 = list_c.index(name_col1), list_c.index(name_col2)
    list_c[idx_2], list_c[idx_1] = list_c[idx_1], list_c[idx_2]
    df = df[list_c]
    return df
