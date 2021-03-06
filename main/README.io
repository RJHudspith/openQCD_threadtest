
********************************************************************************

                            CONFIGURATION I/O

********************************************************************************

On large lattices, the input and output of field configurations may consume a
significant fraction of the available computer time. Parallel I/O should be
used under these conditions, but requires detailed specification in order to
preserve portability across different computer systems and divisions of the
lattice.


TYPES OF I/O PROGRAMS

1. Local I/O

In this case each MPI process writes the field variables on its local lattice
to a configuration file in the order defined by the geometry routines. There
is one configuration file per MPI process. The programs implementing this
form of I/O are write_cnfg() and read_cnfg() (see archive/archive.c).

2. Global I/O

On the MPI process with rank 0, the field variables are collected from all
processes, reordered into a "universal" order and written to a configuration
file. There is a single configuration file. The programs implementing this
form of I/O are export_cnfg() and import_cnfg().

3. Block I/O

In this variant of the I/O operations, the lattice is logically divided into
rectangular blocks of points, which are treated like the full lattice in the
case of the global I/O functions. There is one configuration file per block.
This form of I/O is implemented by blk_export_cnfg() and blk_import_cnfg().


ORDERING OF THE FIELD VARIABLES

The configuration files written by the output functions contain some header
data followed by the gauge-field variables in some order. In all cases, the
program runs through the odd points of certain B0xB1xB2xB3 blocks of lattice
points and sequentially writes the 8 field variables on the links attached in
direction +0,-0,...,+3,-3 to the configuration file(s).

The blocks coincide with the local lattice (type 1), the full lattice (type 2)
or have a specified size (type 3). In the latter case, the blocks must divide
the global lattice and the sizes B0,..,B3 must be integer multiples of the
local lattice sizes L0,..,L3.

The output functions of type 2 and 3 order the lattice points according to
their lexicographic index

 ix=x3+x2*B3+x1*B2*B3+x0*B1*B2*B3,

where (x0,x1,x2,x3) are the Cartesian coordinates of the points relative to
the base point of the block (the point with the smallest global coordinates).
The type 1 output function, on the other hand, runs through the lattice points
in the order defined by the geometry programs.

In all cases, the configuration files are written from the MPI process(es)
containing the base points of the blocks.


HEADER DATA

At the beginning of the configuration files, the output programs write the
following data:

Type 1:  The process grid NPROC0,..,NPROC3, the local lattice sizes L0,..,L3,
         the process block sizes NPROC0_BLK,..,NPROC3_BLK, the coordinates
         cpr[0],..,cpr[3] of the MPI process, the state of the random number
         generators and the local plaquette sum.

Type 2:  The sizes N0,..,N3 of the global lattice and the (global) average
         of the plaquette.

Type 3:  The sizes N0,..,N3 of the global lattice, the block sizes B0,..,B3,
         the coordinates of the base point of the block and the (global)
         average of the plaquette.

Then follow the gauge-field variables as described above.


PARALLEL I/O

Accessing hundreds and thousands of files in parallel can lead to congestions
and consequently to a poor average I/O bandwidth. The program set_nio_streams()
[archive/archive.c] allows to set the number of data streams that may be open
at any given time. Configurations are then written to (and read from) in groups
of that many files.

It is, furthermore, advisable to distribute the possibly very many files
generated by the parallel output functions over many directories. This can be
achieved by choosing the paths of the configuration files passed to the I/O
functions appropriately so that the MPI processes in charge write the data
to files in the desired places.


CONFIGURATION FILE NAMES

The programs in the module archive/archive.c do not assume a particular
file naming convention. It is thus up to the calling program to ensure that
the MPI processes find the correct configuration files.

In the case of the local I/O, for example, the configuration files written
by the MPI processes can be distinguished by appending the process rank as
a suffix to the file basename.

Block configuration files can similarly be labeled by the block index returned
by the program blk_index(). Before reading a configuration, the block sizes
then need to be determined from the configuration file containing the data on
the block with index 0 using the program blk_sizes().

The main programs in the main and devel directories all use the same file
name conventions and configuration directory structure (see README.iodat
in the directory modules/archive).


RECOMMENDATIONS

On machines with local disk drives, the I/O functions of type 1 and 3 achieve
the highest bandwidth, but one risks to loose the configurations if one of the
disks fails at some point and its contents are lost. Instead of the local
disks one can also use external storage systems with these functions, which
may offer a better data protection.

With respect to the type 1 functions, an advantage of the type 2 and type 3
functions is that they produce configuration files in a universal format. In
particular, these can be used on different machines and with different MPI
process grids (as long as the local lattices divide the I/O blocks). Moreover,
these storage formats support periodic and/or anti-periodic extension of an
imported configuration to the full lattice if it was generated on a smaller
lattice.

For I/O of type 3 on machines with multi-core nodes, it is natural to choose
the sizes B0,..,B3 of the logical blocks such that the local lattices hosted
on a given node are all in the same block. In this way the communication
overhead is minimized and the parallel I/O streams are guaranteed to flow
from different nodes.
