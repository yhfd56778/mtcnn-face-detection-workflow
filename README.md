Technical Article:  
Understanding MTCNN: A Practical Guide to Multi-Task Cascaded Convolutional Networks  
https://medium.com/@ace.lin0121/understanding-mtcnn-a-practical-guide-to-multi-task-cascaded-convolutional-networks-for-face-efc6cc2a433f


MTCNN Face Detection Workflow
=============================

This repository provides a public reference implementation and documentation of the MTCNN
(Multi-task Cascaded Convolutional Networks) face detection and alignment workflow.

It includes a PyTorch version of the MTCNN pipeline and supporting notes explaining the
end-to-end detection stages.

Purpose
-------

The goal of this repository is to make the MTCNN workflow and the related technical notes
publicly accessible for reference and verification. All materials in this repository are
intended for learning, reproducibility, and documentation.

Repository Contents
-------------------

MTCNN.py  
A PyTorch-based implementation of the MTCNN detection pipeline, including:
- Proposal generation using P-Net
- Refinement stage using R-Net
- Final stage using O-Net for face classification and landmark prediction
- Basic bounding box handling and workflow structure

Theory and Explanation of MTCNN.docx  
A technical document summarizing:
- The overall architecture of MTCNN
- How P-Net, R-Net and O-Net work together
- Concepts such as bounding box regression and non-maximum suppression
- Landmark refinement and multi-stage inference flow

README.md  
This file.

Current Status
--------------

This repository currently provides:
- Documentation describing how the MTCNN stages operate
- A reference PyTorch script showing the high-level workflow

This repository does not include datasets or full training scripts.

Planned Additions
-----------------

Future updates may include:
- Additional notes or documentation
- Example input and output illustrations

Versioning
----------

v1.0.0 – Initial public release (documentation and reference code)

Contact
-------

For questions or clarifications, please open an Issue on GitHub.
