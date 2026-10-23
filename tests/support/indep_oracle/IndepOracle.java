import java.io.*;
import java.util.*;
import org.openscience.cdk.interfaces.*;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.smiles.SmilesParser;
import io.github.dan2097.jnainchi.*;
import uk.ac.cam.ch.wwmm.opsin.*;

/**
 * A reader of SMILES that shares no code with Orthonym or RDKit: CDK's SmilesParser (the OpenSMILES reading,
 * the lone pair as an implicit hydrogen in every position) hands its atoms, bonds and stereo elements to the
 * InChI library (jna-inchi, bundled in the OPSIN jar) as 0D stereo, and the standard InChIKey comes out.
 * Reads from stdin, one request per line, "smiles TAB name [TAB r]" (r: OPSIN reads the name with radicals allowed,
 * as it does for an input with radical electrons); prints one line "inputKey TAB nameKey" per
 * request. The key of the name is OPSIN's own (NameToInchi), with no SMILES in between.
 *
 * Parity convention (EVEN for CLOCKWISE / OPPOSITE): the tests of tests/unit/test_indep_oracle.py pin it on
 * L-alanine, (E)-but-2-ene, (13C)methane, (S)-methyl phenyl sulfoxide and an N-aryl aziridine, each against
 * OPSIN's own key of the name. Not read: As, Sb, Bi, Te, Po lone-pair centres (the key lacks their parity),
 * aromatic bonds CDK does not kekulise, anything but single, double and triple bonds.
 */
public class IndepOracle {
  static final String TP = "EVEN";   // parity for CLOCKWISE
  static final String DP = "EVEN";   // parity for OPPOSITE
  public static void main(String[] args) throws Exception {
    SmilesParser sp = new SmilesParser(SilentChemObjectBuilder.getInstance());
    NameToStructure n2s = NameToStructure.getInstance();
    BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    String line;
    while ((line = br.readLine()) != null) {
      String[] f = line.split("\t", -1);
      String k1, k2 = "";
      try { k1 = key(sp.parseSmiles(f[0])); } catch (Exception e) { k1 = "ERR:" + e.getClass().getSimpleName(); }
      if (f.length > 1 && !f[1].isEmpty()) {
        try {
          NameToStructureConfig cfg = new NameToStructureConfig();
          cfg.setAllowRadicals(f.length > 2 && f[2].equals("r"));      // a radical input: a name with a free valence
          OpsinResult r = n2s.parseChemicalName(f[1], cfg); k2 = NameToInchi.convertResultToStdInChIKey(r); if (k2 == null) k2 = "NOPARSE"; }
        catch (Exception e) { k2 = "ERR"; }
      }
      System.out.println("KEYS\t" + k1 + "\t" + k2);
      System.out.flush();
    }
  }
  static String key(IAtomContainer m) {
    InchiInput in = new InchiInput();
    Map<IAtom, InchiAtom> map = new IdentityHashMap<>();
    for (IAtom at : m.atoms()) {
      InchiAtom ia = new InchiAtom(at.getSymbol());
      ia.setImplicitHydrogen(at.getImplicitHydrogenCount() == null ? 0 : at.getImplicitHydrogenCount());
      ia.setCharge(at.getFormalCharge() == null ? 0 : at.getFormalCharge());
      if (at.getMassNumber() != null) ia.setIsotopicMass(at.getMassNumber());
      in.addAtom(ia); map.put(at, ia);
    }
    for (IBond b : m.bonds()) {
      InchiBondType t;
      switch (b.getOrder()) { case SINGLE: t = InchiBondType.SINGLE; break; case DOUBLE: t = InchiBondType.DOUBLE; break; case TRIPLE: t = InchiBondType.TRIPLE; break; default: throw new RuntimeException("order"); }
      in.addBond(new InchiBond(map.get(b.getBegin()), map.get(b.getEnd()), t));
    }
    for (IStereoElement se : m.stereoElements()) {
      if (se instanceof ITetrahedralChirality) {
        ITetrahedralChirality tc = (ITetrahedralChirality) se;
        IAtom fa = tc.getChiralAtom(); IAtom[] L = tc.getLigands(); InchiAtom[] n = new InchiAtom[4];
        for (int i = 0; i < 4; i++) n[i] = (L[i] == fa) ? map.get(fa) : map.get(L[i]);
        boolean cw = tc.getStereo() == ITetrahedralChirality.Stereo.CLOCKWISE;
        InchiStereoParity p = InchiStereoParity.valueOf(cw ? TP : (TP.equals("ODD") ? "EVEN" : "ODD"));
        in.addStereo(InchiStereo.createTetrahedralStereo(map.get(fa), n[0], n[1], n[2], n[3], p));
      } else if (se instanceof IDoubleBondStereochemistry) {
        IDoubleBondStereochemistry db = (IDoubleBondStereochemistry) se;
        IBond sb = db.getStereoBond(); IBond[] lig = db.getBonds();
        IAtom a = sb.getBegin(), b = sb.getEnd();
        IAtom x = lig[0].contains(a) ? lig[0].getOther(a) : lig[0].getOther(b);
        IAtom y = lig[1].contains(b) ? lig[1].getOther(b) : lig[1].getOther(a);
        if (!lig[0].contains(a)) { IAtom t = a; a = b; b = t; }
        boolean opp = db.getStereo() == IDoubleBondStereochemistry.Conformation.OPPOSITE;
        InchiStereoParity p = InchiStereoParity.valueOf(opp ? DP : (DP.equals("ODD") ? "EVEN" : "ODD"));
        in.addStereo(InchiStereo.createDoubleBondStereo(map.get(x), map.get(a), map.get(b), map.get(y), p));
      } else throw new RuntimeException("stereo " + se.getClass());
    }
    InchiOutput out = JnaInchi.toInchi(in);
    if (out.getInchi() == null || out.getInchi().isEmpty()) return "ERR:" + out.getMessage();
    return JnaInchi.inchiToInchiKey(out.getInchi()).getInchiKey();
  }
}
