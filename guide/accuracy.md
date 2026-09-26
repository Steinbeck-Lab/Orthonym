# How accuracy is measured

Orthonym is judged by what it gets **exactly right** and by what it gets **wrong**:

- **Round-trip exact match.** Name the structure, parse the name with OPSIN, and compare the full
  InChIKey with the input's. Every molecule counts; a declined molecule counts as a miss.
- **Wrong structures.** Any emitted name that parses to a different molecule. The design target is zero.
- **Preferred-name conformance.** Names checked character for character against molecules the
  Blue Book itself marks as the preferred name.

Benchmark results will be published with the accompanying paper.
