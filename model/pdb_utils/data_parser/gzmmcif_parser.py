import os
import gzip
import warnings

from Bio.File import as_handle
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.Atom import Atom

import numpy

class GZMMCIFParser:
    """Parse a mmCIF file and return a Structure object."""

    def __init__(self, structure_builder=None, QUIET=False):
        """Create a PDBParser object.

        The mmCIF parser calls a number of standard methods in an aggregated
        StructureBuilder object. Normally this object is instanciated by the
        MMCIParser object itself, but if the user provides his/her own
        StructureBuilder object, the latter is used instead.

        Arguments:
         - structure_builder - an optional user implemented StructureBuilder class.
         - QUIET - Evaluated as a Boolean. If true, warnings issued in constructing
           the SMCRA data will be suppressed. If false (DEFAULT), they will be shown.
           These warnings might be indicative of problems in the mmCIF file!

        """
        if structure_builder is not None:
            self._structure_builder = structure_builder
        else:
            self._structure_builder = StructureBuilder()
        self.header = None
        # self.trailer = None
        self.line_counter = 0
        self.build_structure = None
        self.QUIET = bool(QUIET)

    # Public methods

    def get_structure(self, structure_id, filename):
        """Return the structure.

        Arguments:
         - structure_id - string, the id that will be used for the structure
         - filename - name of mmCIF file, OR an open text mode file handle

        """
        with warnings.catch_warnings():
            if self.QUIET:
                warnings.filterwarnings("ignore", category=PDBConstructionWarning)
            self._mmcif_dict = GZMMCIF2Dict(filename)
            self._build_structure(structure_id)
            self._structure_builder.set_header(self._get_header())

        return self._structure_builder.get_structure()

    # Private methods

    def _mmcif_get(self, key, dict, deflt):
        if key in dict:
            rslt = dict[key][0]
            if "?" != rslt:
                return rslt
        return deflt

    def _update_header_entry(self, target_key, keys):
        md = self._mmcif_dict
        for key in keys:
            val = md.get(key)
            try:
                item = val[0]
            except (TypeError, IndexError):
                continue
            if item != "?":
                self.header[target_key] = item
                break

    def _get_header(self):
        self.header = {
            "name": "",
            "head": "",
            "idcode": "",
            "deposition_date": "",
            "structure_method": "",
            "resolution": None,
        }

        self._update_header_entry(
            "idcode", ["_entry_id", "_exptl.entry_id", "_struct.entry_id"]
        )
        self._update_header_entry("name", ["_struct.title"])
        self._update_header_entry(
            "head", ["_struct_keywords.pdbx_keywords", "_struct_keywords.text"]
        )
        self._update_header_entry(
            "deposition_date", ["_pdbx_database_status.recvd_initial_deposition_date"]
        )
        self._update_header_entry("structure_method", ["_exptl.method"])
        self._update_header_entry(
            "resolution",
            [
                "_refine.ls_d_res_high",
                "_refine_hist.d_res_high",
                "_em_3d_reconstruction.resolution",
            ],
        )
        if self.header["resolution"] is not None:
            try:
                self.header["resolution"] = float(self.header["resolution"])
            except ValueError:
                self.header["resolution"] = None

        return self.header

    def _build_structure(self, structure_id):

        # two special chars as placeholders in the mmCIF format
        # for item values that cannot be explicitly assigned
        # see: pdbx/mmcif syntax web page
        _unassigned = {".", "?"}

        mmcif_dict = self._mmcif_dict

        atom_serial_list = mmcif_dict["_atom_site.id"]
        atom_id_list = mmcif_dict["_atom_site.label_atom_id"]
        residue_id_list = mmcif_dict["_atom_site.label_comp_id"]
        try:
            element_list = mmcif_dict["_atom_site.type_symbol"]
        except KeyError:
            element_list = None
        chain_id_list = mmcif_dict["_atom_site.auth_asym_id"]
        x_list = [float(x) for x in mmcif_dict["_atom_site.Cartn_x"]]
        y_list = [float(x) for x in mmcif_dict["_atom_site.Cartn_y"]]
        z_list = [float(x) for x in mmcif_dict["_atom_site.Cartn_z"]]
        alt_list = mmcif_dict["_atom_site.label_alt_id"]
        icode_list = mmcif_dict["_atom_site.pdbx_PDB_ins_code"]
        b_factor_list = mmcif_dict["_atom_site.B_iso_or_equiv"]
        occupancy_list = mmcif_dict["_atom_site.occupancy"]
        fieldname_list = mmcif_dict["_atom_site.group_PDB"]
        try:
            serial_list = [int(n) for n in mmcif_dict["_atom_site.pdbx_PDB_model_num"]]
        except KeyError:
            # No model number column
            serial_list = None
        except ValueError:
            # Invalid model number (malformed file)
            raise PDBConstructionException("Invalid model number") from None
        try:
            aniso_u11 = mmcif_dict["_atom_site_anisotrop.U[1][1]"]
            aniso_u12 = mmcif_dict["_atom_site_anisotrop.U[1][2]"]
            aniso_u13 = mmcif_dict["_atom_site_anisotrop.U[1][3]"]
            aniso_u22 = mmcif_dict["_atom_site_anisotrop.U[2][2]"]
            aniso_u23 = mmcif_dict["_atom_site_anisotrop.U[2][3]"]
            aniso_u33 = mmcif_dict["_atom_site_anisotrop.U[3][3]"]
            aniso_flag = 1
        except KeyError:
            # no anisotropic B factors
            aniso_flag = 0
        # if auth_seq_id is present, we use this.
        # Otherwise label_seq_id is used.
        if "_atom_site.auth_seq_id" in mmcif_dict:
            seq_id_list = mmcif_dict["_atom_site.auth_seq_id"]
        else:
            seq_id_list = mmcif_dict["_atom_site.label_seq_id"]
        # Now loop over atoms and build the structure
        current_chain_id = None
        current_residue_id = None
        current_resname = None
        structure_builder = self._structure_builder
        structure_builder.init_structure(structure_id)
        structure_builder.init_seg(" ")
        # Historically, Biopython PDB parser uses model_id to mean array index
        # so serial_id means the Model ID specified in the file
        current_model_id = -1
        current_serial_id = -1
        for i in range(0, len(atom_id_list)):

            # set the line_counter for 'ATOM' lines only and not
            # as a global line counter found in the PDBParser()
            structure_builder.set_line_counter(i)

            # Try coercing serial to int, for compatibility with PDBParser
            # But do not quit if it fails. mmCIF format specs allow strings.
            try:
                serial = int(atom_serial_list[i])
            except ValueError:
                serial = atom_serial_list[i]
                warnings.warn(
                    "PDBConstructionWarning: "
                    "Some atom serial numbers are not numerical",
                    PDBConstructionWarning,
                )

            x = x_list[i]
            y = y_list[i]
            z = z_list[i]
            resname = residue_id_list[i]
            chainid = chain_id_list[i]
            altloc = alt_list[i]
            if altloc in _unassigned:
                altloc = " "
            int_resseq = int(seq_id_list[i])
            icode = icode_list[i]
            if icode in _unassigned:
                icode = " "
            name = atom_id_list[i]
            # occupancy & B factor
            try:
                tempfactor = float(b_factor_list[i])
            except ValueError:
                raise PDBConstructionException("Invalid or missing B factor") from None
            try:
                occupancy = float(occupancy_list[i])
            except ValueError:
                raise PDBConstructionException("Invalid or missing occupancy") from None
            fieldname = fieldname_list[i]
            if fieldname == "HETATM":
                if resname == "HOH" or resname == "WAT":
                    hetatm_flag = "W"
                else:
                    hetatm_flag = "H"
            else:
                hetatm_flag = " "

            resseq = (hetatm_flag, int_resseq, icode)

            if serial_list is not None:
                # model column exists; use it
                serial_id = serial_list[i]
                if current_serial_id != serial_id:
                    # if serial changes, update it and start new model
                    current_serial_id = serial_id
                    current_model_id += 1
                    structure_builder.init_model(current_model_id, current_serial_id)
                    current_chain_id = None
                    current_residue_id = None
                    current_resname = None
            else:
                # no explicit model column; initialize single model
                structure_builder.init_model(current_model_id)

            if current_chain_id != chainid:
                current_chain_id = chainid
                structure_builder.init_chain(current_chain_id)
                current_residue_id = None
                current_resname = None

            if current_residue_id != resseq or current_resname != resname:
                current_residue_id = resseq
                current_resname = resname
                structure_builder.init_residue(resname, hetatm_flag, int_resseq, icode)

            coord = numpy.array((x, y, z), "f")
            element = element_list[i].upper() if element_list else None
            structure_builder.init_atom(
                name,
                coord,
                tempfactor,
                occupancy,
                altloc,
                name,
                serial_number=serial,
                element=element,
            )
            if aniso_flag == 1 and i < len(aniso_u11):
                u = (
                    aniso_u11[i],
                    aniso_u12[i],
                    aniso_u13[i],
                    aniso_u22[i],
                    aniso_u23[i],
                    aniso_u33[i],
                )
                mapped_anisou = [float(_) for _ in u]
                anisou_array = numpy.array(mapped_anisou, "f")
                structure_builder.set_anisou(anisou_array)
        # Now try to set the cell
        try:
            a = float(mmcif_dict["_cell.length_a"][0])
            b = float(mmcif_dict["_cell.length_b"][0])
            c = float(mmcif_dict["_cell.length_c"][0])
            alpha = float(mmcif_dict["_cell.angle_alpha"][0])
            beta = float(mmcif_dict["_cell.angle_beta"][0])
            gamma = float(mmcif_dict["_cell.angle_gamma"][0])
            cell = numpy.array((a, b, c, alpha, beta, gamma), "f")
            spacegroup = mmcif_dict["_symmetry.space_group_name_H-M"][0]
            spacegroup = spacegroup[1:-1]  # get rid of quotes!!
            if spacegroup is None:
                raise Exception
            structure_builder.set_symmetry(spacegroup, cell)
        except Exception:
            pass  # no cell found, so just ignore


class GZMMCIF2Dict(dict):
    """Parse a mmCIF file and return a dictionary."""

    def __init__(self, filename):
        """Parse a mmCIF file and return a dictionary.

        Arguments:
         - file - name of the PDB file OR an open filehandle

        """
        self.quote_chars = ["'", '"']
        self.whitespace_chars = [" ", "\t"]
        with gzip.open(filename, 'rt') as handle:
            loop_flag = False
            key = None
            tokens = self._tokenize(handle)
            try:
                token = next(tokens)
            except StopIteration:
                return  # for Python 3.7 and PEP 479
            self[token[0:5]] = token[5:]
            i = 0
            n = 0
            for token in tokens:
                if token.lower() == "loop_":
                    loop_flag = True
                    keys = []
                    i = 0
                    n = 0
                    continue
                elif loop_flag:
                    # The second condition checks we are in the first column
                    # Some mmCIF files (e.g. 4q9r) have values in later columns
                    # starting with an underscore and we don't want to read
                    # these as keys
                    if token.startswith("_") and (n == 0 or i % n == 0):
                        if i > 0:
                            loop_flag = False
                        else:
                            self[token] = []
                            keys.append(token)
                            n += 1
                            continue
                    else:
                        self[keys[i % n]].append(token)
                        i += 1
                        continue
                if key is None:
                    key = token
                else:
                    self[key] = [token]
                    key = None

    # Private methods

    def _splitline(self, line):
        # See https://www.iucr.org/resources/cif/spec/version1.1/cifsyntax for the syntax
        in_token = False
        # quote character of the currently open quote, or None if no quote open
        quote_open_char = None
        start_i = 0
        for (i, c) in enumerate(line):
            if c in self.whitespace_chars:
                if in_token and not quote_open_char:
                    in_token = False
                    yield line[start_i:i]
            elif c in self.quote_chars:
                if not quote_open_char and not in_token:
                    quote_open_char = c
                    in_token = True
                    start_i = i + 1
                elif c == quote_open_char and (
                    i + 1 == len(line) or line[i + 1] in self.whitespace_chars
                ):
                    quote_open_char = None
                    in_token = False
                    yield line[start_i:i]
            elif c == "#" and not in_token:
                # Skip comments. "#" is a valid non-comment char inside of a
                # quote and inside of an unquoted token (!?!?), so we need to
                # check that the current char is not in a token.
                return
            elif not in_token:
                in_token = True
                start_i = i
        if in_token:
            yield line[start_i:]
        if quote_open_char:
            raise ValueError("Line ended with quote open: " + line)

    def _tokenize(self, handle):
        empty = True
        for line in handle:
            empty = False
            # bytes.startswith()
            if line.startswith("#"):
                continue
            elif line.startswith(";"):
                # The spec says that leading whitespace on each line must be
                # preserved while trailing whitespace may be stripped.  The
                # trailing newline must be stripped.
                token_buffer = [line[1:].rstrip()]
                for line in handle:
                    line = line.rstrip()
                    if line.startswith(";"):
                        yield "\n".join(token_buffer)
                        line = line[1:]
                        if line and not line[0] in self.whitespace_chars:
                            raise ValueError("Missing whitespace")
                        break
                    token_buffer.append(line)
                else:
                    raise ValueError("Missing closing semicolon")
            yield from self._splitline(line.strip())
        if empty:
            raise ValueError("Empty file.")