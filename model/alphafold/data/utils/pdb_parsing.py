import logging
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from alphafold.common import residue_constants

logger= logging.getLogger(__name__)


def get_pdb_all_atoms(pdb_file):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('target', pdb_file)

    model_structures = []
    for model in structure:
        model_structure = {}
        for chain in model:
            chain_id = chain.id
            sequence = [residue_constants.restype_3to1[res.get_resname()] for res in chain]
            sequence = ''.join(sequence)

            seq_len = len(sequence)
            chain_all_atom_positions = np.zeros((seq_len, 37, 3))
            chain_all_atom_mask = np.zeros((seq_len, 37))

            for res_idx, residue in enumerate(chain):
                for atom in residue:
                    atom_type = atom.get_name()
                    coord = atom.get_coord()
                    try:
                        atom_index = residue_constants.atom_order[atom_type]
                        chain_all_atom_mask[res_idx, atom_index] = 1
                        chain_all_atom_positions[res_idx, atom_index] = coord
                    except:
                        # logger.debug(f'ignore atom {atom_type}')
                        pass
            
            model_structure[chain_id] = {
                "all_atom_positions": chain_all_atom_positions,
                "all_atom_mask": chain_all_atom_mask,
            }
        model_structures.append(model_structure)
    return model_structures
                

