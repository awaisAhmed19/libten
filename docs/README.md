# libten Documentation

`libten` is a standalone C11 tensor library for machine learning and numerical computation, including real and complex arithmetic and a CPU reference backend with CUDA support when available.

- `DESIGN.md`: architecture and subsystem boundaries.
- `MATHEMATICS.md`: shape, stride, views, broadcasting, complex arithmetic, reductions, and matmul semantics.
- `DECISIONS.md`: binding decisions and unresolved design questions.
- `BACKLOG.md`: implementation milestones and acceptance criteria.
- `STRUCTURE.md`: directory and file-by-file responsibilities.

Keep these docs synchronized: decisions in `DECISIONS.md`, architecture in `DESIGN.md`, math contracts in `MATHEMATICS.md`, progress in `BACKLOG.md`, and file ownership in `STRUCTURE.md`. These documents describe intended design, not necessarily already-implemented behavior.
