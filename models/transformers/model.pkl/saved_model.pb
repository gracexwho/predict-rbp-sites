??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:
*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:
*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	? *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
?
3transformer_block/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *D
shared_name53transformer_block/multi_head_attention/query/kernel
?
Gtransformer_block/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp3transformer_block/multi_head_attention/query/kernel*"
_output_shapes
:  *
dtype0
?
1transformer_block/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *B
shared_name31transformer_block/multi_head_attention/query/bias
?
Etransformer_block/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp1transformer_block/multi_head_attention/query/bias*
_output_shapes

: *
dtype0
?
1transformer_block/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *B
shared_name31transformer_block/multi_head_attention/key/kernel
?
Etransformer_block/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp1transformer_block/multi_head_attention/key/kernel*"
_output_shapes
:  *
dtype0
?
/transformer_block/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *@
shared_name1/transformer_block/multi_head_attention/key/bias
?
Ctransformer_block/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_attention/key/bias*
_output_shapes

: *
dtype0
?
3transformer_block/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *D
shared_name53transformer_block/multi_head_attention/value/kernel
?
Gtransformer_block/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp3transformer_block/multi_head_attention/value/kernel*"
_output_shapes
:  *
dtype0
?
1transformer_block/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *B
shared_name31transformer_block/multi_head_attention/value/bias
?
Etransformer_block/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp1transformer_block/multi_head_attention/value/bias*
_output_shapes

: *
dtype0
?
>transformer_block/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *O
shared_name@>transformer_block/multi_head_attention/attention_output/kernel
?
Rtransformer_block/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp>transformer_block/multi_head_attention/attention_output/kernel*"
_output_shapes
:  *
dtype0
?
<transformer_block/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><transformer_block/multi_head_attention/attention_output/bias
?
Ptransformer_block/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp<transformer_block/multi_head_attention/attention_output/bias*
_output_shapes
: *
dtype0
?
transformer_block/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  */
shared_name transformer_block/dense/kernel
?
2transformer_block/dense/kernel/Read/ReadVariableOpReadVariableOptransformer_block/dense/kernel*
_output_shapes

:  *
dtype0
?
transformer_block/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nametransformer_block/dense/bias
?
0transformer_block/dense/bias/Read/ReadVariableOpReadVariableOptransformer_block/dense/bias*
_output_shapes
: *
dtype0
?
 transformer_block/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *1
shared_name" transformer_block/dense_1/kernel
?
4transformer_block/dense_1/kernel/Read/ReadVariableOpReadVariableOp transformer_block/dense_1/kernel*
_output_shapes

:  *
dtype0
?
transformer_block/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name transformer_block/dense_1/bias
?
2transformer_block/dense_1/bias/Read/ReadVariableOpReadVariableOptransformer_block/dense_1/bias*
_output_shapes
: *
dtype0
?
+transformer_block/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+transformer_block/layer_normalization/gamma
?
?transformer_block/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp+transformer_block/layer_normalization/gamma*
_output_shapes
: *
dtype0
?
*transformer_block/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*transformer_block/layer_normalization/beta
?
>transformer_block/layer_normalization/beta/Read/ReadVariableOpReadVariableOp*transformer_block/layer_normalization/beta*
_output_shapes
: *
dtype0
?
-transformer_block/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-transformer_block/layer_normalization_1/gamma
?
Atransformer_block/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp-transformer_block/layer_normalization_1/gamma*
_output_shapes
: *
dtype0
?
,transformer_block/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,transformer_block/layer_normalization_1/beta
?
@transformer_block/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp,transformer_block/layer_normalization_1/beta*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
RMSprop/conv1d/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nameRMSprop/conv1d/kernel/rms
?
-RMSprop/conv1d/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1d/kernel/rms*"
_output_shapes
:
*
dtype0
?
RMSprop/conv1d/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameRMSprop/conv1d/bias/rms

+RMSprop/conv1d/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1d/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/conv1d_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameRMSprop/conv1d_1/kernel/rms
?
/RMSprop/conv1d_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1d_1/kernel/rms*"
_output_shapes
:
*
dtype0
?
RMSprop/conv1d_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/conv1d_1/bias/rms
?
-RMSprop/conv1d_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1d_1/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/dense_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *+
shared_nameRMSprop/dense_2/kernel/rms
?
.RMSprop/dense_2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/kernel/rms*
_output_shapes
:	? *
dtype0
?
RMSprop/dense_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameRMSprop/dense_2/bias/rms
?
,RMSprop/dense_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/dense_3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nameRMSprop/dense_3/kernel/rms
?
.RMSprop/dense_3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/kernel/rms*
_output_shapes

: *
dtype0
?
RMSprop/dense_3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_3/bias/rms
?
,RMSprop/dense_3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/bias/rms*
_output_shapes
:*
dtype0
?
?RMSprop/transformer_block/multi_head_attention/query/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *P
shared_nameA?RMSprop/transformer_block/multi_head_attention/query/kernel/rms
?
SRMSprop/transformer_block/multi_head_attention/query/kernel/rms/Read/ReadVariableOpReadVariableOp?RMSprop/transformer_block/multi_head_attention/query/kernel/rms*"
_output_shapes
:  *
dtype0
?
=RMSprop/transformer_block/multi_head_attention/query/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *N
shared_name?=RMSprop/transformer_block/multi_head_attention/query/bias/rms
?
QRMSprop/transformer_block/multi_head_attention/query/bias/rms/Read/ReadVariableOpReadVariableOp=RMSprop/transformer_block/multi_head_attention/query/bias/rms*
_output_shapes

: *
dtype0
?
=RMSprop/transformer_block/multi_head_attention/key/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *N
shared_name?=RMSprop/transformer_block/multi_head_attention/key/kernel/rms
?
QRMSprop/transformer_block/multi_head_attention/key/kernel/rms/Read/ReadVariableOpReadVariableOp=RMSprop/transformer_block/multi_head_attention/key/kernel/rms*"
_output_shapes
:  *
dtype0
?
;RMSprop/transformer_block/multi_head_attention/key/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *L
shared_name=;RMSprop/transformer_block/multi_head_attention/key/bias/rms
?
ORMSprop/transformer_block/multi_head_attention/key/bias/rms/Read/ReadVariableOpReadVariableOp;RMSprop/transformer_block/multi_head_attention/key/bias/rms*
_output_shapes

: *
dtype0
?
?RMSprop/transformer_block/multi_head_attention/value/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *P
shared_nameA?RMSprop/transformer_block/multi_head_attention/value/kernel/rms
?
SRMSprop/transformer_block/multi_head_attention/value/kernel/rms/Read/ReadVariableOpReadVariableOp?RMSprop/transformer_block/multi_head_attention/value/kernel/rms*"
_output_shapes
:  *
dtype0
?
=RMSprop/transformer_block/multi_head_attention/value/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *N
shared_name?=RMSprop/transformer_block/multi_head_attention/value/bias/rms
?
QRMSprop/transformer_block/multi_head_attention/value/bias/rms/Read/ReadVariableOpReadVariableOp=RMSprop/transformer_block/multi_head_attention/value/bias/rms*
_output_shapes

: *
dtype0
?
JRMSprop/transformer_block/multi_head_attention/attention_output/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *[
shared_nameLJRMSprop/transformer_block/multi_head_attention/attention_output/kernel/rms
?
^RMSprop/transformer_block/multi_head_attention/attention_output/kernel/rms/Read/ReadVariableOpReadVariableOpJRMSprop/transformer_block/multi_head_attention/attention_output/kernel/rms*"
_output_shapes
:  *
dtype0
?
HRMSprop/transformer_block/multi_head_attention/attention_output/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Y
shared_nameJHRMSprop/transformer_block/multi_head_attention/attention_output/bias/rms
?
\RMSprop/transformer_block/multi_head_attention/attention_output/bias/rms/Read/ReadVariableOpReadVariableOpHRMSprop/transformer_block/multi_head_attention/attention_output/bias/rms*
_output_shapes
: *
dtype0
?
*RMSprop/transformer_block/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *;
shared_name,*RMSprop/transformer_block/dense/kernel/rms
?
>RMSprop/transformer_block/dense/kernel/rms/Read/ReadVariableOpReadVariableOp*RMSprop/transformer_block/dense/kernel/rms*
_output_shapes

:  *
dtype0
?
(RMSprop/transformer_block/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(RMSprop/transformer_block/dense/bias/rms
?
<RMSprop/transformer_block/dense/bias/rms/Read/ReadVariableOpReadVariableOp(RMSprop/transformer_block/dense/bias/rms*
_output_shapes
: *
dtype0
?
,RMSprop/transformer_block/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *=
shared_name.,RMSprop/transformer_block/dense_1/kernel/rms
?
@RMSprop/transformer_block/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOp,RMSprop/transformer_block/dense_1/kernel/rms*
_output_shapes

:  *
dtype0
?
*RMSprop/transformer_block/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*RMSprop/transformer_block/dense_1/bias/rms
?
>RMSprop/transformer_block/dense_1/bias/rms/Read/ReadVariableOpReadVariableOp*RMSprop/transformer_block/dense_1/bias/rms*
_output_shapes
: *
dtype0
?
7RMSprop/transformer_block/layer_normalization/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97RMSprop/transformer_block/layer_normalization/gamma/rms
?
KRMSprop/transformer_block/layer_normalization/gamma/rms/Read/ReadVariableOpReadVariableOp7RMSprop/transformer_block/layer_normalization/gamma/rms*
_output_shapes
: *
dtype0
?
6RMSprop/transformer_block/layer_normalization/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86RMSprop/transformer_block/layer_normalization/beta/rms
?
JRMSprop/transformer_block/layer_normalization/beta/rms/Read/ReadVariableOpReadVariableOp6RMSprop/transformer_block/layer_normalization/beta/rms*
_output_shapes
: *
dtype0
?
9RMSprop/transformer_block/layer_normalization_1/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9RMSprop/transformer_block/layer_normalization_1/gamma/rms
?
MRMSprop/transformer_block/layer_normalization_1/gamma/rms/Read/ReadVariableOpReadVariableOp9RMSprop/transformer_block/layer_normalization_1/gamma/rms*
_output_shapes
: *
dtype0
?
8RMSprop/transformer_block/layer_normalization_1/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8RMSprop/transformer_block/layer_normalization_1/beta/rms
?
LRMSprop/transformer_block/layer_normalization_1/beta/rms/Read/ReadVariableOpReadVariableOp8RMSprop/transformer_block/layer_normalization_1/beta/rms*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ڄ
valueτB˄ BÄ
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
layer_with_weights-4
layer-13
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
R
!trainable_variables
"regularization_losses
#	variables
$	keras_api
R
%trainable_variables
&regularization_losses
'	variables
(	keras_api
R
)trainable_variables
*regularization_losses
+	variables
,	keras_api
R
-trainable_variables
.regularization_losses
/	variables
0	keras_api
R
1trainable_variables
2regularization_losses
3	variables
4	keras_api
?
5att
6ffn1
7ffn2
8
layernorm1
9
layernorm2
:dropout1
;dropout2
<trainable_variables
=regularization_losses
>	variables
?	keras_api
R
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
R
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
h

Hkernel
Ibias
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
h

Nkernel
Obias
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
?
Titer
	Udecay
Vlearning_rate
Wmomentum
Xrho
rms?
rms?
rms?
rms?
Hrms?
Irms?
Nrms?
Orms?
Yrms?
Zrms?
[rms?
\rms?
]rms?
^rms?
_rms?
`rms?
arms?
brms?
crms?
drms?
erms?
frms?
grms?
hrms?
?
0
1
2
3
Y4
Z5
[6
\7
]8
^9
_10
`11
a12
b13
c14
d15
e16
f17
g18
h19
H20
I21
N22
O23
 
?
0
1
2
3
Y4
Z5
[6
\7
]8
^9
_10
`11
a12
b13
c14
d15
e16
f17
g18
h19
H20
I21
N22
O23
?
trainable_variables
regularization_losses
	variables
ilayer_metrics
jlayer_regularization_losses
kmetrics

llayers
mnon_trainable_variables
 
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
regularization_losses
	variables
nlayer_metrics
olayer_regularization_losses
pmetrics

qlayers
rnon_trainable_variables
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
regularization_losses
	variables
slayer_metrics
tlayer_regularization_losses
umetrics

vlayers
wnon_trainable_variables
 
 
 
?
!trainable_variables
"regularization_losses
#	variables
xlayer_metrics
ylayer_regularization_losses
zmetrics

{layers
|non_trainable_variables
 
 
 
?
%trainable_variables
&regularization_losses
'	variables
}layer_metrics
~layer_regularization_losses
metrics
?layers
?non_trainable_variables
 
 
 
?
)trainable_variables
*regularization_losses
+	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 
 
?
-trainable_variables
.regularization_losses
/	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 
 
?
1trainable_variables
2regularization_losses
3	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?
?_query_dense
?
_key_dense
?_value_dense
?_softmax
?_dropout_layer
?_output_dense
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

akernel
bbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

ckernel
dbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
v
	?axis
	egamma
fbeta
?trainable_variables
?regularization_losses
?	variables
?	keras_api
v
	?axis
	ggamma
hbeta
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
v
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
 
v
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
?
<trainable_variables
=regularization_losses
>	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 
 
?
@trainable_variables
Aregularization_losses
B	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 
 
?
Dtrainable_variables
Eregularization_losses
F	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
 

H0
I1
?
Jtrainable_variables
Kregularization_losses
L	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
 

N0
O1
?
Ptrainable_variables
Qregularization_losses
R	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE3transformer_block/multi_head_attention/query/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1transformer_block/multi_head_attention/query/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1transformer_block/multi_head_attention/key/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/transformer_block/multi_head_attention/key/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE3transformer_block/multi_head_attention/value/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1transformer_block/multi_head_attention/value/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>transformer_block/multi_head_attention/attention_output/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<transformer_block/multi_head_attention/attention_output/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtransformer_block/dense/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtransformer_block/dense/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE transformer_block/dense_1/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtransformer_block/dense_1/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE+transformer_block/layer_normalization/gamma1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*transformer_block/layer_normalization/beta1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-transformer_block/layer_normalization_1/gamma1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,transformer_block/layer_normalization_1/beta1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
?partial_output_shape
?full_output_shape

Ykernel
Zbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?partial_output_shape
?full_output_shape

[kernel
\bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?partial_output_shape
?full_output_shape

]kernel
^bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?partial_output_shape
?full_output_shape

_kernel
`bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
8
Y0
Z1
[2
\3
]4
^5
_6
`7
 
8
Y0
Z1
[2
\3
]4
^5
_6
`7
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables

a0
b1
 

a0
b1
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables

c0
d1
 

c0
d1
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 

e0
f1
 

e0
f1
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 

g0
h1
 

g0
h1
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 
 
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 
 
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 
 
1
50
61
72
83
94
:5
;6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 

Y0
Z1
 

Y0
Z1
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 

[0
\1
 

[0
\1
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 

]0
^1
 

]0
^1
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 
 
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 
 
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 

_0
`1
 

_0
`1
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
 
 
 
0
?0
?1
?2
?3
?4
?5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
??
VARIABLE_VALUERMSprop/conv1d/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/conv1d/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv1d_1/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/conv1d_1/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense_2/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUERMSprop/dense_2/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense_3/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUERMSprop/dense_3/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?RMSprop/transformer_block/multi_head_attention/query/kernel/rmsNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=RMSprop/transformer_block/multi_head_attention/query/bias/rmsNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=RMSprop/transformer_block/multi_head_attention/key/kernel/rmsNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;RMSprop/transformer_block/multi_head_attention/key/bias/rmsNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?RMSprop/transformer_block/multi_head_attention/value/kernel/rmsNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=RMSprop/transformer_block/multi_head_attention/value/bias/rmsNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEJRMSprop/transformer_block/multi_head_attention/attention_output/kernel/rmsOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEHRMSprop/transformer_block/multi_head_attention/attention_output/bias/rmsOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*RMSprop/transformer_block/dense/kernel/rmsOtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(RMSprop/transformer_block/dense/bias/rmsOtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,RMSprop/transformer_block/dense_1/kernel/rmsOtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*RMSprop/transformer_block/dense_1/bias/rmsOtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7RMSprop/transformer_block/layer_normalization/gamma/rmsOtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6RMSprop/transformer_block/layer_normalization/beta/rmsOtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9RMSprop/transformer_block/layer_normalization_1/gamma/rmsOtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8RMSprop/transformer_block/layer_normalization_1/beta/rmsOtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
p
serving_default_input_1Placeholder*"
_output_shapes
:2o*
dtype0*
shape:2o
p
serving_default_input_2Placeholder*"
_output_shapes
:2o*
dtype0*
shape:2o
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv1d_1/kernelconv1d_1/biasconv1d/kernelconv1d/bias3transformer_block/multi_head_attention/query/kernel1transformer_block/multi_head_attention/query/bias1transformer_block/multi_head_attention/key/kernel/transformer_block/multi_head_attention/key/bias3transformer_block/multi_head_attention/value/kernel1transformer_block/multi_head_attention/value/bias>transformer_block/multi_head_attention/attention_output/kernel<transformer_block/multi_head_attention/attention_output/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/betatransformer_block/dense/kerneltransformer_block/dense/bias transformer_block/dense_1/kerneltransformer_block/dense_1/bias-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betadense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_17122
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOpGtransformer_block/multi_head_attention/query/kernel/Read/ReadVariableOpEtransformer_block/multi_head_attention/query/bias/Read/ReadVariableOpEtransformer_block/multi_head_attention/key/kernel/Read/ReadVariableOpCtransformer_block/multi_head_attention/key/bias/Read/ReadVariableOpGtransformer_block/multi_head_attention/value/kernel/Read/ReadVariableOpEtransformer_block/multi_head_attention/value/bias/Read/ReadVariableOpRtransformer_block/multi_head_attention/attention_output/kernel/Read/ReadVariableOpPtransformer_block/multi_head_attention/attention_output/bias/Read/ReadVariableOp2transformer_block/dense/kernel/Read/ReadVariableOp0transformer_block/dense/bias/Read/ReadVariableOp4transformer_block/dense_1/kernel/Read/ReadVariableOp2transformer_block/dense_1/bias/Read/ReadVariableOp?transformer_block/layer_normalization/gamma/Read/ReadVariableOp>transformer_block/layer_normalization/beta/Read/ReadVariableOpAtransformer_block/layer_normalization_1/gamma/Read/ReadVariableOp@transformer_block/layer_normalization_1/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-RMSprop/conv1d/kernel/rms/Read/ReadVariableOp+RMSprop/conv1d/bias/rms/Read/ReadVariableOp/RMSprop/conv1d_1/kernel/rms/Read/ReadVariableOp-RMSprop/conv1d_1/bias/rms/Read/ReadVariableOp.RMSprop/dense_2/kernel/rms/Read/ReadVariableOp,RMSprop/dense_2/bias/rms/Read/ReadVariableOp.RMSprop/dense_3/kernel/rms/Read/ReadVariableOp,RMSprop/dense_3/bias/rms/Read/ReadVariableOpSRMSprop/transformer_block/multi_head_attention/query/kernel/rms/Read/ReadVariableOpQRMSprop/transformer_block/multi_head_attention/query/bias/rms/Read/ReadVariableOpQRMSprop/transformer_block/multi_head_attention/key/kernel/rms/Read/ReadVariableOpORMSprop/transformer_block/multi_head_attention/key/bias/rms/Read/ReadVariableOpSRMSprop/transformer_block/multi_head_attention/value/kernel/rms/Read/ReadVariableOpQRMSprop/transformer_block/multi_head_attention/value/bias/rms/Read/ReadVariableOp^RMSprop/transformer_block/multi_head_attention/attention_output/kernel/rms/Read/ReadVariableOp\RMSprop/transformer_block/multi_head_attention/attention_output/bias/rms/Read/ReadVariableOp>RMSprop/transformer_block/dense/kernel/rms/Read/ReadVariableOp<RMSprop/transformer_block/dense/bias/rms/Read/ReadVariableOp@RMSprop/transformer_block/dense_1/kernel/rms/Read/ReadVariableOp>RMSprop/transformer_block/dense_1/bias/rms/Read/ReadVariableOpKRMSprop/transformer_block/layer_normalization/gamma/rms/Read/ReadVariableOpJRMSprop/transformer_block/layer_normalization/beta/rms/Read/ReadVariableOpMRMSprop/transformer_block/layer_normalization_1/gamma/rms/Read/ReadVariableOpLRMSprop/transformer_block/layer_normalization_1/beta/rms/Read/ReadVariableOpConst*D
Tin=
;29	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_18239
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rho3transformer_block/multi_head_attention/query/kernel1transformer_block/multi_head_attention/query/bias1transformer_block/multi_head_attention/key/kernel/transformer_block/multi_head_attention/key/bias3transformer_block/multi_head_attention/value/kernel1transformer_block/multi_head_attention/value/bias>transformer_block/multi_head_attention/attention_output/kernel<transformer_block/multi_head_attention/attention_output/biastransformer_block/dense/kerneltransformer_block/dense/bias transformer_block/dense_1/kerneltransformer_block/dense_1/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/beta-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betatotalcountRMSprop/conv1d/kernel/rmsRMSprop/conv1d/bias/rmsRMSprop/conv1d_1/kernel/rmsRMSprop/conv1d_1/bias/rmsRMSprop/dense_2/kernel/rmsRMSprop/dense_2/bias/rmsRMSprop/dense_3/kernel/rmsRMSprop/dense_3/bias/rms?RMSprop/transformer_block/multi_head_attention/query/kernel/rms=RMSprop/transformer_block/multi_head_attention/query/bias/rms=RMSprop/transformer_block/multi_head_attention/key/kernel/rms;RMSprop/transformer_block/multi_head_attention/key/bias/rms?RMSprop/transformer_block/multi_head_attention/value/kernel/rms=RMSprop/transformer_block/multi_head_attention/value/bias/rmsJRMSprop/transformer_block/multi_head_attention/attention_output/kernel/rmsHRMSprop/transformer_block/multi_head_attention/attention_output/bias/rms*RMSprop/transformer_block/dense/kernel/rms(RMSprop/transformer_block/dense/bias/rms,RMSprop/transformer_block/dense_1/kernel/rms*RMSprop/transformer_block/dense_1/bias/rms7RMSprop/transformer_block/layer_normalization/gamma/rms6RMSprop/transformer_block/layer_normalization/beta/rms9RMSprop/transformer_block/layer_normalization_1/gamma/rms8RMSprop/transformer_block/layer_normalization_1/beta/rms*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_18414??
?
|
'__inference_dense_3_layer_call_fn_18050

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_167372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*%
_input_shapes
:2 ::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:2 
 
_user_specified_nameinputs
˷
?
L__inference_transformer_block_layer_call_and_return_conditional_losses_17803

inputsD
@multi_head_attention_query_einsum_einsum_readvariableop_resource:
6multi_head_attention_query_add_readvariableop_resourceB
>multi_head_attention_key_einsum_einsum_readvariableop_resource8
4multi_head_attention_key_add_readvariableop_resourceD
@multi_head_attention_value_einsum_einsum_readvariableop_resource:
6multi_head_attention_value_add_readvariableop_resourceO
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resourceE
Amulti_head_attention_attention_output_add_readvariableop_resource=
9layer_normalization_batchnorm_mul_readvariableop_resource9
5layer_normalization_batchnorm_readvariableop_resource+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource?
;layer_normalization_1_batchnorm_mul_readvariableop_resource;
7layer_normalization_1_batchnorm_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention/query/add/ReadVariableOp?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2 
multi_head_attention/query/add?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOp?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsum?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype02-
+multi_head_attention/key/add/ReadVariableOp?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2
multi_head_attention/key/add?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention/value/add/ReadVariableOp?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention/Mul/y?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*&
_output_shapes
:2" 2
multi_head_attention/Mul?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*&
_output_shapes
:2""*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/Einsum?
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*&
_output_shapes
:2""2&
$multi_head_attention/softmax/Softmax?
*multi_head_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*multi_head_attention/dropout/dropout/Const?
(multi_head_attention/dropout/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:03multi_head_attention/dropout/dropout/Const:output:0*
T0*&
_output_shapes
:2""2*
(multi_head_attention/dropout/dropout/Mul?
*multi_head_attention/dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"2      "   "   2,
*multi_head_attention/dropout/dropout/Shape?
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform3multi_head_attention/dropout/dropout/Shape:output:0*
T0*&
_output_shapes
:2""*
dtype02C
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniform?
3multi_head_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3multi_head_attention/dropout/dropout/GreaterEqual/y?
1multi_head_attention/dropout/dropout/GreaterEqualGreaterEqualJmulti_head_attention/dropout/dropout/random_uniform/RandomUniform:output:0<multi_head_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:2""23
1multi_head_attention/dropout/dropout/GreaterEqual?
)multi_head_attention/dropout/dropout/CastCast5multi_head_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*&
_output_shapes
:2""2+
)multi_head_attention/dropout/dropout/Cast?
*multi_head_attention/dropout/dropout/Mul_1Mul,multi_head_attention/dropout/dropout/Mul:z:0-multi_head_attention/dropout/dropout/Cast:y:0*
T0*&
_output_shapes
:2""2,
*multi_head_attention/dropout/dropout/Mul_1?
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*&
_output_shapes
:2" *
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/Einsum?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*"
_output_shapes
:2" *
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsum?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOp?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2+
)multi_head_attention/attention_output/addw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMul-multi_head_attention/attention_output/add:z:0 dropout_2/dropout/Const:output:0*
T0*"
_output_shapes
:2" 2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*"
_output_shapes
:2" *
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2" 2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2" 2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*"
_output_shapes
:2" 2
dropout_2/dropout/Mul_1e
addAddV2inputsdropout_2/dropout/Mul_1:z:0*
T0*"
_output_shapes
:2" 2
add?
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2"
 layer_normalization/moments/mean?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*"
_output_shapes
:2"2*
(layer_normalization/moments/StopGradient?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 2/
-layer_normalization/moments/SquaredDifference?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2&
$layer_normalization/moments/variance?
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52%
#layer_normalization/batchnorm/add/y?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"2#
!layer_normalization/batchnorm/add?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*"
_output_shapes
:2"2%
#layer_normalization/batchnorm/Rsqrt?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2#
!layer_normalization/batchnorm/mul?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/mul_1?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/mul_2?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,layer_normalization/batchnorm/ReadVariableOp?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 2#
!layer_normalization/batchnorm/sub?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/add_1?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense/Tensordot/ReadVariableOp?
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      2
dense/Tensordot/Reshape/shape?
dense/Tensordot/ReshapeReshape'layer_normalization/batchnorm/add_1:z:0&dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense/Tensordot/MatMul?
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dense/Tensordot/shape?
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2
dense/BiasAdde

dense/ReluReludense/BiasAdd:output:0*
T0*"
_output_shapes
:2" 2

dense/Relu?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_1/Tensordot/ReadVariableOp?
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      2!
dense_1/Tensordot/Reshape/shape?
dense_1/Tensordot/ReshapeReshapedense/Relu:activations:0(dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2
dense_1/Tensordot/Reshape?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_1/Tensordot/MatMul?
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dense_1/Tensordot/shape?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2
dense_1/Tensordot?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2
dense_1/BiasAddw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMuldense_1/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*"
_output_shapes
:2" 2
dropout_3/dropout/Mul?
dropout_3/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*"
_output_shapes
:2" *
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2" 2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2" 2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*"
_output_shapes
:2" 2
dropout_3/dropout/Mul_1?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_3/dropout/Mul_1:z:0*
T0*"
_output_shapes
:2" 2
add_1?
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2$
"layer_normalization_1/moments/mean?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:2"2,
*layer_normalization_1/moments/StopGradient?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 21
/layer_normalization_1/moments/SquaredDifference?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2(
&layer_normalization_1/moments/variance?
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_1/batchnorm/add/y?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"2%
#layer_normalization_1/batchnorm/add?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*"
_output_shapes
:2"2'
%layer_normalization_1/batchnorm/Rsqrt?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization_1/batchnorm/mul?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/mul_1?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/mul_2?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization_1/batchnorm/sub?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/add_1?
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*
T0*"
_output_shapes
:2" 2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:2" ::::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:J F
"
_output_shapes
:2" 
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_16182

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_17462
inputs_0
inputs_18
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resourceV
Rtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resourceL
Htransformer_block_multi_head_attention_query_add_readvariableop_resourceT
Ptransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resourceJ
Ftransformer_block_multi_head_attention_key_add_readvariableop_resourceV
Rtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resourceL
Htransformer_block_multi_head_attention_value_add_readvariableop_resourcea
]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resourceW
Stransformer_block_multi_head_attention_attention_output_add_readvariableop_resourceO
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resourceK
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resource=
9transformer_block_dense_tensordot_readvariableop_resource;
7transformer_block_dense_biasadd_readvariableop_resource?
;transformer_block_dense_1_tensordot_readvariableop_resource=
9transformer_block_dense_1_biasadd_readvariableop_resourceQ
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resourceM
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity??conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?.transformer_block/dense/BiasAdd/ReadVariableOp?0transformer_block/dense/Tensordot/ReadVariableOp?0transformer_block/dense_1/BiasAdd/ReadVariableOp?2transformer_block/dense_1/Tensordot/ReadVariableOp?>transformer_block/layer_normalization/batchnorm/ReadVariableOp?Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp?Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp?Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?=transformer_block/multi_head_attention/key/add/ReadVariableOp?Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp??transformer_block/multi_head_attention/query/add/ReadVariableOp?Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp??transformer_block/multi_head_attention/value/add/ReadVariableOp?Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsinputs_1'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2o2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*&
_output_shapes
:2f*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*"
_output_shapes
:2f*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2f2
conv1d_1/BiasAddn
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*"
_output_shapes
:2f2
conv1d_1/Relu?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsinputs_0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2o2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*&
_output_shapes
:2f*
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*"
_output_shapes
:2f*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2f2
conv1d/BiasAddh
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*"
_output_shapes
:2f2
conv1d/Relu?
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim?
max_pooling1d_1/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2f2
max_pooling1d_1/ExpandDims?
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*&
_output_shapes
:2"*
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPool?
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*"
_output_shapes
:2"*
squeeze_dims
2
max_pooling1d_1/Squeeze~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2f2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*&
_output_shapes
:2"*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*"
_output_shapes
:2"*
squeeze_dims
2
max_pooling1d/Squeeze}
dropout/IdentityIdentitymax_pooling1d/Squeeze:output:0*
T0*"
_output_shapes
:2"2
dropout/Identity?
dropout_1/IdentityIdentity max_pooling1d_1/Squeeze:output:0*
T0*"
_output_shapes
:2"2
dropout_1/Identityt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2dropout/Identity:output:0dropout_1/Identity:output:0 concatenate/concat/axis:output:0*
N*
T0*"
_output_shapes
:2" 2
concatenate/concat?
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02K
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp?
:transformer_block/multi_head_attention/query/einsum/EinsumEinsumconcatenate/concat:output:0Qtransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2<
:transformer_block/multi_head_attention/query/einsum/Einsum?
?transformer_block/multi_head_attention/query/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype02A
?transformer_block/multi_head_attention/query/add/ReadVariableOp?
0transformer_block/multi_head_attention/query/addAddV2Ctransformer_block/multi_head_attention/query/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 22
0transformer_block/multi_head_attention/query/add?
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02I
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp?
8transformer_block/multi_head_attention/key/einsum/EinsumEinsumconcatenate/concat:output:0Otransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2:
8transformer_block/multi_head_attention/key/einsum/Einsum?
=transformer_block/multi_head_attention/key/add/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype02?
=transformer_block/multi_head_attention/key/add/ReadVariableOp?
.transformer_block/multi_head_attention/key/addAddV2Atransformer_block/multi_head_attention/key/einsum/Einsum:output:0Etransformer_block/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 20
.transformer_block/multi_head_attention/key/add?
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02K
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp?
:transformer_block/multi_head_attention/value/einsum/EinsumEinsumconcatenate/concat:output:0Qtransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2<
:transformer_block/multi_head_attention/value/einsum/Einsum?
?transformer_block/multi_head_attention/value/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype02A
?transformer_block/multi_head_attention/value/add/ReadVariableOp?
0transformer_block/multi_head_attention/value/addAddV2Ctransformer_block/multi_head_attention/value/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 22
0transformer_block/multi_head_attention/value/add?
,transformer_block/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2.
,transformer_block/multi_head_attention/Mul/y?
*transformer_block/multi_head_attention/MulMul4transformer_block/multi_head_attention/query/add:z:05transformer_block/multi_head_attention/Mul/y:output:0*
T0*&
_output_shapes
:2" 2,
*transformer_block/multi_head_attention/Mul?
4transformer_block/multi_head_attention/einsum/EinsumEinsum2transformer_block/multi_head_attention/key/add:z:0.transformer_block/multi_head_attention/Mul:z:0*
N*
T0*&
_output_shapes
:2""*
equationaecd,abcd->acbe26
4transformer_block/multi_head_attention/einsum/Einsum?
6transformer_block/multi_head_attention/softmax/SoftmaxSoftmax=transformer_block/multi_head_attention/einsum/Einsum:output:0*
T0*&
_output_shapes
:2""28
6transformer_block/multi_head_attention/softmax/Softmax?
7transformer_block/multi_head_attention/dropout/IdentityIdentity@transformer_block/multi_head_attention/softmax/Softmax:softmax:0*
T0*&
_output_shapes
:2""29
7transformer_block/multi_head_attention/dropout/Identity?
6transformer_block/multi_head_attention/einsum_1/EinsumEinsum@transformer_block/multi_head_attention/dropout/Identity:output:04transformer_block/multi_head_attention/value/add:z:0*
N*
T0*&
_output_shapes
:2" *
equationacbe,aecd->abcd28
6transformer_block/multi_head_attention/einsum_1/Einsum?
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02V
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?
Etransformer_block/multi_head_attention/attention_output/einsum/EinsumEinsum?transformer_block/multi_head_attention/einsum_1/Einsum:output:0\transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*"
_output_shapes
:2" *
equationabcd,cde->abe2G
Etransformer_block/multi_head_attention/attention_output/einsum/Einsum?
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02L
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp?
;transformer_block/multi_head_attention/attention_output/addAddV2Ntransformer_block/multi_head_attention/attention_output/einsum/Einsum:output:0Rtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2=
;transformer_block/multi_head_attention/attention_output/add?
$transformer_block/dropout_2/IdentityIdentity?transformer_block/multi_head_attention/attention_output/add:z:0*
T0*"
_output_shapes
:2" 2&
$transformer_block/dropout_2/Identity?
transformer_block/addAddV2concatenate/concat:output:0-transformer_block/dropout_2/Identity:output:0*
T0*"
_output_shapes
:2" 2
transformer_block/add?
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2F
Dtransformer_block/layer_normalization/moments/mean/reduction_indices?
2transformer_block/layer_normalization/moments/meanMeantransformer_block/add:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(24
2transformer_block/layer_normalization/moments/mean?
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*
T0*"
_output_shapes
:2"2<
:transformer_block/layer_normalization/moments/StopGradient?
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 2A
?transformer_block/layer_normalization/moments/SquaredDifference?
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block/layer_normalization/moments/variance/reduction_indices?
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(28
6transformer_block/layer_normalization/moments/variance?
5transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?527
5transformer_block/layer_normalization/batchnorm/add/y?
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"25
3transformer_block/layer_normalization/batchnorm/add?
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*"
_output_shapes
:2"27
5transformer_block/layer_normalization/batchnorm/Rsqrt?
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 25
3transformer_block/layer_normalization/batchnorm/mul?
5transformer_block/layer_normalization/batchnorm/mul_1Multransformer_block/add:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 27
5transformer_block/layer_normalization/batchnorm/mul_1?
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 27
5transformer_block/layer_normalization/batchnorm/mul_2?
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02@
>transformer_block/layer_normalization/batchnorm/ReadVariableOp?
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 25
3transformer_block/layer_normalization/batchnorm/sub?
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 27
5transformer_block/layer_normalization/batchnorm/add_1?
0transformer_block/dense/Tensordot/ReadVariableOpReadVariableOp9transformer_block_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype022
0transformer_block/dense/Tensordot/ReadVariableOp?
/transformer_block/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      21
/transformer_block/dense/Tensordot/Reshape/shape?
)transformer_block/dense/Tensordot/ReshapeReshape9transformer_block/layer_normalization/batchnorm/add_1:z:08transformer_block/dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2+
)transformer_block/dense/Tensordot/Reshape?
(transformer_block/dense/Tensordot/MatMulMatMul2transformer_block/dense/Tensordot/Reshape:output:08transformer_block/dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2*
(transformer_block/dense/Tensordot/MatMul?
'transformer_block/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2)
'transformer_block/dense/Tensordot/shape?
!transformer_block/dense/TensordotReshape2transformer_block/dense/Tensordot/MatMul:product:00transformer_block/dense/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2#
!transformer_block/dense/Tensordot?
.transformer_block/dense/BiasAdd/ReadVariableOpReadVariableOp7transformer_block_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.transformer_block/dense/BiasAdd/ReadVariableOp?
transformer_block/dense/BiasAddBiasAdd*transformer_block/dense/Tensordot:output:06transformer_block/dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2!
transformer_block/dense/BiasAdd?
transformer_block/dense/ReluRelu(transformer_block/dense/BiasAdd:output:0*
T0*"
_output_shapes
:2" 2
transformer_block/dense/Relu?
2transformer_block/dense_1/Tensordot/ReadVariableOpReadVariableOp;transformer_block_dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype024
2transformer_block/dense_1/Tensordot/ReadVariableOp?
1transformer_block/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      23
1transformer_block/dense_1/Tensordot/Reshape/shape?
+transformer_block/dense_1/Tensordot/ReshapeReshape*transformer_block/dense/Relu:activations:0:transformer_block/dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2-
+transformer_block/dense_1/Tensordot/Reshape?
*transformer_block/dense_1/Tensordot/MatMulMatMul4transformer_block/dense_1/Tensordot/Reshape:output:0:transformer_block/dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2,
*transformer_block/dense_1/Tensordot/MatMul?
)transformer_block/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2+
)transformer_block/dense_1/Tensordot/shape?
#transformer_block/dense_1/TensordotReshape4transformer_block/dense_1/Tensordot/MatMul:product:02transformer_block/dense_1/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2%
#transformer_block/dense_1/Tensordot?
0transformer_block/dense_1/BiasAdd/ReadVariableOpReadVariableOp9transformer_block_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0transformer_block/dense_1/BiasAdd/ReadVariableOp?
!transformer_block/dense_1/BiasAddBiasAdd,transformer_block/dense_1/Tensordot:output:08transformer_block/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2#
!transformer_block/dense_1/BiasAdd?
$transformer_block/dropout_3/IdentityIdentity*transformer_block/dense_1/BiasAdd:output:0*
T0*"
_output_shapes
:2" 2&
$transformer_block/dropout_3/Identity?
transformer_block/add_1AddV29transformer_block/layer_normalization/batchnorm/add_1:z:0-transformer_block/dropout_3/Identity:output:0*
T0*"
_output_shapes
:2" 2
transformer_block/add_1?
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2H
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indices?
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add_1:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(26
4transformer_block/layer_normalization_1/moments/mean?
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:2"2>
<transformer_block/layer_normalization_1/moments/StopGradient?
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add_1:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 2C
Atransformer_block/layer_normalization_1/moments/SquaredDifference?
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indices?
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2:
8transformer_block/layer_normalization_1/moments/variance?
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?529
7transformer_block/layer_normalization_1/batchnorm/add/y?
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"27
5transformer_block/layer_normalization_1/batchnorm/add?
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*"
_output_shapes
:2"29
7transformer_block/layer_normalization_1/batchnorm/Rsqrt?
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02F
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp?
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 27
5transformer_block/layer_normalization_1/batchnorm/mul?
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add_1:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 29
7transformer_block/layer_normalization_1/batchnorm/mul_1?
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 29
7transformer_block/layer_normalization_1/batchnorm/mul_2?
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 27
5transformer_block/layer_normalization_1/batchnorm/sub?
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 29
7transformer_block/layer_normalization_1/batchnorm/add_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten/Const?
flatten/ReshapeReshape;transformer_block/layer_normalization_1/batchnorm/add_1:z:0flatten/Const:output:0*
T0*
_output_shapes
:	2?2
flatten/Reshapex
dropout_4/IdentityIdentityflatten/Reshape:output:0*
T0*
_output_shapes
:	2?2
dropout_4/Identity?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_4/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2 2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2 2
dense_2/BiasAddg
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*
_output_shapes

:2 2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:22
dense_3/BiasAddp
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*
_output_shapes

:22
dense_3/Softmax?
IdentityIdentitydense_3/Softmax:softmax:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp/^transformer_block/dense/BiasAdd/ReadVariableOp1^transformer_block/dense/Tensordot/ReadVariableOp1^transformer_block/dense_1/BiasAdd/ReadVariableOp3^transformer_block/dense_1/Tensordot/ReadVariableOp?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpK^transformer_block/multi_head_attention/attention_output/add/ReadVariableOpU^transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp>^transformer_block/multi_head_attention/key/add/ReadVariableOpH^transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/query/add/ReadVariableOpJ^transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/value/add/ReadVariableOpJ^transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2`
.transformer_block/dense/BiasAdd/ReadVariableOp.transformer_block/dense/BiasAdd/ReadVariableOp2d
0transformer_block/dense/Tensordot/ReadVariableOp0transformer_block/dense/Tensordot/ReadVariableOp2d
0transformer_block/dense_1/BiasAdd/ReadVariableOp0transformer_block/dense_1/BiasAdd/ReadVariableOp2h
2transformer_block/dense_1/Tensordot/ReadVariableOp2transformer_block/dense_1/Tensordot/ReadVariableOp2?
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2?
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2?
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2?
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpJtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp2?
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpTtransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2~
=transformer_block/multi_head_attention/key/add/ReadVariableOp=transformer_block/multi_head_attention/key/add/ReadVariableOp2?
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpGtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
?transformer_block/multi_head_attention/query/add/ReadVariableOp?transformer_block/multi_head_attention/query/add/ReadVariableOp2?
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
?transformer_block/multi_head_attention/value/add/ReadVariableOp?transformer_block/multi_head_attention/value/add/ReadVariableOp2?
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:L H
"
_output_shapes
:2o
"
_user_specified_name
inputs/0:LH
"
_output_shapes
:2o
"
_user_specified_name
inputs/1
?
W
+__inference_concatenate_layer_call_fn_17687
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_163262
PartitionedCallg
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
:2" 2

Identity"
identityIdentity:output:0*/
_input_shapes
:2":2":L H
"
_output_shapes
:2"
"
_user_specified_name
inputs/0:LH
"
_output_shapes
:2"
"
_user_specified_name
inputs/1
Ø
?
L__inference_transformer_block_layer_call_and_return_conditional_losses_16546

inputsD
@multi_head_attention_query_einsum_einsum_readvariableop_resource:
6multi_head_attention_query_add_readvariableop_resourceB
>multi_head_attention_key_einsum_einsum_readvariableop_resource8
4multi_head_attention_key_add_readvariableop_resourceD
@multi_head_attention_value_einsum_einsum_readvariableop_resource:
6multi_head_attention_value_add_readvariableop_resourceO
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resourceE
Amulti_head_attention_attention_output_add_readvariableop_resource=
9layer_normalization_batchnorm_mul_readvariableop_resource9
5layer_normalization_batchnorm_readvariableop_resource+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource?
;layer_normalization_1_batchnorm_mul_readvariableop_resource;
7layer_normalization_1_batchnorm_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention/query/add/ReadVariableOp?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2 
multi_head_attention/query/add?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOp?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsum?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype02-
+multi_head_attention/key/add/ReadVariableOp?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2
multi_head_attention/key/add?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention/value/add/ReadVariableOp?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention/Mul/y?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*&
_output_shapes
:2" 2
multi_head_attention/Mul?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*&
_output_shapes
:2""*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/Einsum?
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*&
_output_shapes
:2""2&
$multi_head_attention/softmax/Softmax?
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*&
_output_shapes
:2""2'
%multi_head_attention/dropout/Identity?
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*&
_output_shapes
:2" *
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/Einsum?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*"
_output_shapes
:2" *
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsum?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOp?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2+
)multi_head_attention/attention_output/add?
dropout_2/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*"
_output_shapes
:2" 2
dropout_2/Identitye
addAddV2inputsdropout_2/Identity:output:0*
T0*"
_output_shapes
:2" 2
add?
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2"
 layer_normalization/moments/mean?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*"
_output_shapes
:2"2*
(layer_normalization/moments/StopGradient?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 2/
-layer_normalization/moments/SquaredDifference?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2&
$layer_normalization/moments/variance?
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52%
#layer_normalization/batchnorm/add/y?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"2#
!layer_normalization/batchnorm/add?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*"
_output_shapes
:2"2%
#layer_normalization/batchnorm/Rsqrt?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2#
!layer_normalization/batchnorm/mul?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/mul_1?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/mul_2?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,layer_normalization/batchnorm/ReadVariableOp?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 2#
!layer_normalization/batchnorm/sub?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/add_1?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense/Tensordot/ReadVariableOp?
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      2
dense/Tensordot/Reshape/shape?
dense/Tensordot/ReshapeReshape'layer_normalization/batchnorm/add_1:z:0&dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense/Tensordot/MatMul?
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dense/Tensordot/shape?
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2
dense/BiasAdde

dense/ReluReludense/BiasAdd:output:0*
T0*"
_output_shapes
:2" 2

dense/Relu?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_1/Tensordot/ReadVariableOp?
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      2!
dense_1/Tensordot/Reshape/shape?
dense_1/Tensordot/ReshapeReshapedense/Relu:activations:0(dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2
dense_1/Tensordot/Reshape?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_1/Tensordot/MatMul?
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dense_1/Tensordot/shape?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2
dense_1/Tensordot?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2
dense_1/BiasAdd{
dropout_3/IdentityIdentitydense_1/BiasAdd:output:0*
T0*"
_output_shapes
:2" 2
dropout_3/Identity?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_3/Identity:output:0*
T0*"
_output_shapes
:2" 2
add_1?
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2$
"layer_normalization_1/moments/mean?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:2"2,
*layer_normalization_1/moments/StopGradient?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 21
/layer_normalization_1/moments/SquaredDifference?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2(
&layer_normalization_1/moments/variance?
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_1/batchnorm/add/y?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"2%
#layer_normalization_1/batchnorm/add?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*"
_output_shapes
:2"2'
%layer_normalization_1/batchnorm/Rsqrt?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization_1/batchnorm/mul?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/mul_1?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/mul_2?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization_1/batchnorm/sub?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/add_1?
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*
T0*"
_output_shapes
:2" 2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:2" ::::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:J F
"
_output_shapes
:2" 
 
_user_specified_nameinputs
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_16710

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2 2	
BiasAddO
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:2 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2 2

Identity"
identityIdentity:output:0*&
_input_shapes
:	2?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	2?
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_17058
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_170072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:2o
!
_user_specified_name	input_1:KG
"
_output_shapes
:2o
!
_user_specified_name	input_2
?
K
/__inference_max_pooling1d_1_layer_call_fn_16188

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_161822
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_17647

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_162762
PartitionedCallg
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
:2"2

Identity"
identityIdentity:output:0*!
_input_shapes
:2":J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_17978

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Const_
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	2?2	
Reshape\
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	2?2

Identity"
identityIdentity:output:0*!
_input_shapes
:2" :J F
"
_output_shapes
:2" 
 
_user_specified_nameinputs
??
?$
!__inference__traced_restore_18414
file_prefix"
assignvariableop_conv1d_kernel"
assignvariableop_1_conv1d_bias&
"assignvariableop_2_conv1d_1_kernel$
 assignvariableop_3_conv1d_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias#
assignvariableop_8_rmsprop_iter$
 assignvariableop_9_rmsprop_decay-
)assignvariableop_10_rmsprop_learning_rate(
$assignvariableop_11_rmsprop_momentum#
assignvariableop_12_rmsprop_rhoK
Gassignvariableop_13_transformer_block_multi_head_attention_query_kernelI
Eassignvariableop_14_transformer_block_multi_head_attention_query_biasI
Eassignvariableop_15_transformer_block_multi_head_attention_key_kernelG
Cassignvariableop_16_transformer_block_multi_head_attention_key_biasK
Gassignvariableop_17_transformer_block_multi_head_attention_value_kernelI
Eassignvariableop_18_transformer_block_multi_head_attention_value_biasV
Rassignvariableop_19_transformer_block_multi_head_attention_attention_output_kernelT
Passignvariableop_20_transformer_block_multi_head_attention_attention_output_bias6
2assignvariableop_21_transformer_block_dense_kernel4
0assignvariableop_22_transformer_block_dense_bias8
4assignvariableop_23_transformer_block_dense_1_kernel6
2assignvariableop_24_transformer_block_dense_1_biasC
?assignvariableop_25_transformer_block_layer_normalization_gammaB
>assignvariableop_26_transformer_block_layer_normalization_betaE
Aassignvariableop_27_transformer_block_layer_normalization_1_gammaD
@assignvariableop_28_transformer_block_layer_normalization_1_beta
assignvariableop_29_total
assignvariableop_30_count1
-assignvariableop_31_rmsprop_conv1d_kernel_rms/
+assignvariableop_32_rmsprop_conv1d_bias_rms3
/assignvariableop_33_rmsprop_conv1d_1_kernel_rms1
-assignvariableop_34_rmsprop_conv1d_1_bias_rms2
.assignvariableop_35_rmsprop_dense_2_kernel_rms0
,assignvariableop_36_rmsprop_dense_2_bias_rms2
.assignvariableop_37_rmsprop_dense_3_kernel_rms0
,assignvariableop_38_rmsprop_dense_3_bias_rmsW
Sassignvariableop_39_rmsprop_transformer_block_multi_head_attention_query_kernel_rmsU
Qassignvariableop_40_rmsprop_transformer_block_multi_head_attention_query_bias_rmsU
Qassignvariableop_41_rmsprop_transformer_block_multi_head_attention_key_kernel_rmsS
Oassignvariableop_42_rmsprop_transformer_block_multi_head_attention_key_bias_rmsW
Sassignvariableop_43_rmsprop_transformer_block_multi_head_attention_value_kernel_rmsU
Qassignvariableop_44_rmsprop_transformer_block_multi_head_attention_value_bias_rmsb
^assignvariableop_45_rmsprop_transformer_block_multi_head_attention_attention_output_kernel_rms`
\assignvariableop_46_rmsprop_transformer_block_multi_head_attention_attention_output_bias_rmsB
>assignvariableop_47_rmsprop_transformer_block_dense_kernel_rms@
<assignvariableop_48_rmsprop_transformer_block_dense_bias_rmsD
@assignvariableop_49_rmsprop_transformer_block_dense_1_kernel_rmsB
>assignvariableop_50_rmsprop_transformer_block_dense_1_bias_rmsO
Kassignvariableop_51_rmsprop_transformer_block_layer_normalization_gamma_rmsN
Jassignvariableop_52_rmsprop_transformer_block_layer_normalization_beta_rmsQ
Massignvariableop_53_rmsprop_transformer_block_layer_normalization_1_gamma_rmsP
Lassignvariableop_54_rmsprop_transformer_block_layer_normalization_1_beta_rms
identity_56??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_rmsprop_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp)assignvariableop_10_rmsprop_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_rmsprop_momentumIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_rmsprop_rhoIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpGassignvariableop_13_transformer_block_multi_head_attention_query_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpEassignvariableop_14_transformer_block_multi_head_attention_query_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpEassignvariableop_15_transformer_block_multi_head_attention_key_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpCassignvariableop_16_transformer_block_multi_head_attention_key_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpGassignvariableop_17_transformer_block_multi_head_attention_value_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpEassignvariableop_18_transformer_block_multi_head_attention_value_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpRassignvariableop_19_transformer_block_multi_head_attention_attention_output_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpPassignvariableop_20_transformer_block_multi_head_attention_attention_output_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp2assignvariableop_21_transformer_block_dense_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_transformer_block_dense_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_transformer_block_dense_1_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp2assignvariableop_24_transformer_block_dense_1_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp?assignvariableop_25_transformer_block_layer_normalization_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp>assignvariableop_26_transformer_block_layer_normalization_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpAassignvariableop_27_transformer_block_layer_normalization_1_gammaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp@assignvariableop_28_transformer_block_layer_normalization_1_betaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp-assignvariableop_31_rmsprop_conv1d_kernel_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_rmsprop_conv1d_bias_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp/assignvariableop_33_rmsprop_conv1d_1_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp-assignvariableop_34_rmsprop_conv1d_1_bias_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp.assignvariableop_35_rmsprop_dense_2_kernel_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp,assignvariableop_36_rmsprop_dense_2_bias_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp.assignvariableop_37_rmsprop_dense_3_kernel_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp,assignvariableop_38_rmsprop_dense_3_bias_rmsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpSassignvariableop_39_rmsprop_transformer_block_multi_head_attention_query_kernel_rmsIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpQassignvariableop_40_rmsprop_transformer_block_multi_head_attention_query_bias_rmsIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpQassignvariableop_41_rmsprop_transformer_block_multi_head_attention_key_kernel_rmsIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpOassignvariableop_42_rmsprop_transformer_block_multi_head_attention_key_bias_rmsIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpSassignvariableop_43_rmsprop_transformer_block_multi_head_attention_value_kernel_rmsIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpQassignvariableop_44_rmsprop_transformer_block_multi_head_attention_value_bias_rmsIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp^assignvariableop_45_rmsprop_transformer_block_multi_head_attention_attention_output_kernel_rmsIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp\assignvariableop_46_rmsprop_transformer_block_multi_head_attention_attention_output_bias_rmsIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp>assignvariableop_47_rmsprop_transformer_block_dense_kernel_rmsIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp<assignvariableop_48_rmsprop_transformer_block_dense_bias_rmsIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp@assignvariableop_49_rmsprop_transformer_block_dense_1_kernel_rmsIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp>assignvariableop_50_rmsprop_transformer_block_dense_1_bias_rmsIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpKassignvariableop_51_rmsprop_transformer_block_layer_normalization_gamma_rmsIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpJassignvariableop_52_rmsprop_transformer_block_layer_normalization_beta_rmsIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpMassignvariableop_53_rmsprop_transformer_block_layer_normalization_1_gamma_rmsIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpLassignvariableop_54_rmsprop_transformer_block_layer_normalization_1_beta_rmsIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_549
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55?

Identity_56IdentityIdentity_55:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_56"#
identity_56Identity_56:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_16306

inputs

identity_1U
IdentityIdentityinputs*
T0*"
_output_shapes
:2"2

Identityd

Identity_1IdentityIdentity:output:0*
T0*"
_output_shapes
:2"2

Identity_1"!

identity_1Identity_1:output:0*!
_input_shapes
:2":J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_17122
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_161582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:2o
!
_user_specified_name	input_1:KG
"
_output_shapes
:2o
!
_user_specified_name	input_2
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_16686

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2?2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2?2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	2?:G C

_output_shapes
:	2?
 
_user_specified_nameinputs
?

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_17995

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2?2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"2   @  2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	2?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2?2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2?2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2?2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2?2

Identity"
identityIdentity:output:0*
_input_shapes
:	2?:G C

_output_shapes
:	2?
 
_user_specified_nameinputs
?
{
&__inference_conv1d_layer_call_fn_17595

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_162412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:2f2

Identity"
identityIdentity:output:0*)
_input_shapes
:2o::22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:2o
 
_user_specified_nameinputs
?
b
)__inference_dropout_4_layer_call_fn_18005

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_166812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2?2

Identity"
identityIdentity:output:0*
_input_shapes
:	2?22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	2?
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_17669

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_163012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:2"2

Identity"
identityIdentity:output:0*!
_input_shapes
:2"22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_17674

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_163062
PartitionedCallg
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
:2"2

Identity"
identityIdentity:output:0*!
_input_shapes
:2":J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_16167

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_17642

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_162712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:2"2

Identity"
identityIdentity:output:0*!
_input_shapes
:2"22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
?

a
B__inference_dropout_layer_call_and_return_conditional_losses_17632

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constn
dropout/MulMulinputsdropout/Const:output:0*
T0*"
_output_shapes
:2"2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "      2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*"
_output_shapes
:2"*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2"2
dropout/GreaterEqualz
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2"2
dropout/Castu
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*"
_output_shapes
:2"2
dropout/Mul_1`
IdentityIdentitydropout/Mul_1:z:0*
T0*"
_output_shapes
:2"2

Identity"
identityIdentity:output:0*!
_input_shapes
:2":J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
Ø
?
L__inference_transformer_block_layer_call_and_return_conditional_losses_17898

inputsD
@multi_head_attention_query_einsum_einsum_readvariableop_resource:
6multi_head_attention_query_add_readvariableop_resourceB
>multi_head_attention_key_einsum_einsum_readvariableop_resource8
4multi_head_attention_key_add_readvariableop_resourceD
@multi_head_attention_value_einsum_einsum_readvariableop_resource:
6multi_head_attention_value_add_readvariableop_resourceO
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resourceE
Amulti_head_attention_attention_output_add_readvariableop_resource=
9layer_normalization_batchnorm_mul_readvariableop_resource9
5layer_normalization_batchnorm_readvariableop_resource+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource?
;layer_normalization_1_batchnorm_mul_readvariableop_resource;
7layer_normalization_1_batchnorm_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention/query/add/ReadVariableOp?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2 
multi_head_attention/query/add?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOp?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsum?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype02-
+multi_head_attention/key/add/ReadVariableOp?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2
multi_head_attention/key/add?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention/value/add/ReadVariableOp?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention/Mul/y?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*&
_output_shapes
:2" 2
multi_head_attention/Mul?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*&
_output_shapes
:2""*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/Einsum?
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*&
_output_shapes
:2""2&
$multi_head_attention/softmax/Softmax?
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*&
_output_shapes
:2""2'
%multi_head_attention/dropout/Identity?
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*&
_output_shapes
:2" *
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/Einsum?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*"
_output_shapes
:2" *
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsum?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOp?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2+
)multi_head_attention/attention_output/add?
dropout_2/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*"
_output_shapes
:2" 2
dropout_2/Identitye
addAddV2inputsdropout_2/Identity:output:0*
T0*"
_output_shapes
:2" 2
add?
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2"
 layer_normalization/moments/mean?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*"
_output_shapes
:2"2*
(layer_normalization/moments/StopGradient?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 2/
-layer_normalization/moments/SquaredDifference?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2&
$layer_normalization/moments/variance?
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52%
#layer_normalization/batchnorm/add/y?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"2#
!layer_normalization/batchnorm/add?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*"
_output_shapes
:2"2%
#layer_normalization/batchnorm/Rsqrt?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2#
!layer_normalization/batchnorm/mul?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/mul_1?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/mul_2?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,layer_normalization/batchnorm/ReadVariableOp?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 2#
!layer_normalization/batchnorm/sub?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/add_1?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense/Tensordot/ReadVariableOp?
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      2
dense/Tensordot/Reshape/shape?
dense/Tensordot/ReshapeReshape'layer_normalization/batchnorm/add_1:z:0&dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense/Tensordot/MatMul?
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dense/Tensordot/shape?
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2
dense/BiasAdde

dense/ReluReludense/BiasAdd:output:0*
T0*"
_output_shapes
:2" 2

dense/Relu?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_1/Tensordot/ReadVariableOp?
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      2!
dense_1/Tensordot/Reshape/shape?
dense_1/Tensordot/ReshapeReshapedense/Relu:activations:0(dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2
dense_1/Tensordot/Reshape?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_1/Tensordot/MatMul?
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dense_1/Tensordot/shape?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2
dense_1/Tensordot?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2
dense_1/BiasAdd{
dropout_3/IdentityIdentitydense_1/BiasAdd:output:0*
T0*"
_output_shapes
:2" 2
dropout_3/Identity?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_3/Identity:output:0*
T0*"
_output_shapes
:2" 2
add_1?
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2$
"layer_normalization_1/moments/mean?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:2"2,
*layer_normalization_1/moments/StopGradient?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 21
/layer_normalization_1/moments/SquaredDifference?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2(
&layer_normalization_1/moments/variance?
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_1/batchnorm/add/y?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"2%
#layer_normalization_1/batchnorm/add?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*"
_output_shapes
:2"2'
%layer_normalization_1/batchnorm/Rsqrt?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization_1/batchnorm/mul?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/mul_1?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/mul_2?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization_1/batchnorm/sub?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/add_1?
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*
T0*"
_output_shapes
:2" 2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:2" ::::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:J F
"
_output_shapes
:2" 
 
_user_specified_nameinputs
?
I
-__inference_max_pooling1d_layer_call_fn_16173

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_161672
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
˷
?
L__inference_transformer_block_layer_call_and_return_conditional_losses_16451

inputsD
@multi_head_attention_query_einsum_einsum_readvariableop_resource:
6multi_head_attention_query_add_readvariableop_resourceB
>multi_head_attention_key_einsum_einsum_readvariableop_resource8
4multi_head_attention_key_add_readvariableop_resourceD
@multi_head_attention_value_einsum_einsum_readvariableop_resource:
6multi_head_attention_value_add_readvariableop_resourceO
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resourceE
Amulti_head_attention_attention_output_add_readvariableop_resource=
9layer_normalization_batchnorm_mul_readvariableop_resource9
5layer_normalization_batchnorm_readvariableop_resource+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource?
;layer_normalization_1_batchnorm_mul_readvariableop_resource;
7layer_normalization_1_batchnorm_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention/query/add/ReadVariableOp?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2 
multi_head_attention/query/add?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOp?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsum?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype02-
+multi_head_attention/key/add/ReadVariableOp?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2
multi_head_attention/key/add?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention/value/add/ReadVariableOp?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention/Mul/y?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*&
_output_shapes
:2" 2
multi_head_attention/Mul?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*&
_output_shapes
:2""*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/Einsum?
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*&
_output_shapes
:2""2&
$multi_head_attention/softmax/Softmax?
*multi_head_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*multi_head_attention/dropout/dropout/Const?
(multi_head_attention/dropout/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:03multi_head_attention/dropout/dropout/Const:output:0*
T0*&
_output_shapes
:2""2*
(multi_head_attention/dropout/dropout/Mul?
*multi_head_attention/dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"2      "   "   2,
*multi_head_attention/dropout/dropout/Shape?
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform3multi_head_attention/dropout/dropout/Shape:output:0*
T0*&
_output_shapes
:2""*
dtype02C
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniform?
3multi_head_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3multi_head_attention/dropout/dropout/GreaterEqual/y?
1multi_head_attention/dropout/dropout/GreaterEqualGreaterEqualJmulti_head_attention/dropout/dropout/random_uniform/RandomUniform:output:0<multi_head_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:2""23
1multi_head_attention/dropout/dropout/GreaterEqual?
)multi_head_attention/dropout/dropout/CastCast5multi_head_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*&
_output_shapes
:2""2+
)multi_head_attention/dropout/dropout/Cast?
*multi_head_attention/dropout/dropout/Mul_1Mul,multi_head_attention/dropout/dropout/Mul:z:0-multi_head_attention/dropout/dropout/Cast:y:0*
T0*&
_output_shapes
:2""2,
*multi_head_attention/dropout/dropout/Mul_1?
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*&
_output_shapes
:2" *
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/Einsum?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*"
_output_shapes
:2" *
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsum?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOp?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2+
)multi_head_attention/attention_output/addw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMul-multi_head_attention/attention_output/add:z:0 dropout_2/dropout/Const:output:0*
T0*"
_output_shapes
:2" 2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*"
_output_shapes
:2" *
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2" 2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2" 2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*"
_output_shapes
:2" 2
dropout_2/dropout/Mul_1e
addAddV2inputsdropout_2/dropout/Mul_1:z:0*
T0*"
_output_shapes
:2" 2
add?
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2"
 layer_normalization/moments/mean?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*"
_output_shapes
:2"2*
(layer_normalization/moments/StopGradient?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 2/
-layer_normalization/moments/SquaredDifference?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2&
$layer_normalization/moments/variance?
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52%
#layer_normalization/batchnorm/add/y?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"2#
!layer_normalization/batchnorm/add?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*"
_output_shapes
:2"2%
#layer_normalization/batchnorm/Rsqrt?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2#
!layer_normalization/batchnorm/mul?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/mul_1?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/mul_2?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,layer_normalization/batchnorm/ReadVariableOp?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 2#
!layer_normalization/batchnorm/sub?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization/batchnorm/add_1?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense/Tensordot/ReadVariableOp?
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      2
dense/Tensordot/Reshape/shape?
dense/Tensordot/ReshapeReshape'layer_normalization/batchnorm/add_1:z:0&dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense/Tensordot/MatMul?
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dense/Tensordot/shape?
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2
dense/BiasAdde

dense/ReluReludense/BiasAdd:output:0*
T0*"
_output_shapes
:2" 2

dense/Relu?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_1/Tensordot/ReadVariableOp?
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      2!
dense_1/Tensordot/Reshape/shape?
dense_1/Tensordot/ReshapeReshapedense/Relu:activations:0(dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2
dense_1/Tensordot/Reshape?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_1/Tensordot/MatMul?
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dense_1/Tensordot/shape?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2
dense_1/Tensordot?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2
dense_1/BiasAddw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMuldense_1/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*"
_output_shapes
:2" 2
dropout_3/dropout/Mul?
dropout_3/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*"
_output_shapes
:2" *
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2" 2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2" 2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*"
_output_shapes
:2" 2
dropout_3/dropout/Mul_1?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_3/dropout/Mul_1:z:0*
T0*"
_output_shapes
:2" 2
add_1?
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2$
"layer_normalization_1/moments/mean?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:2"2,
*layer_normalization_1/moments/StopGradient?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 21
/layer_normalization_1/moments/SquaredDifference?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2(
&layer_normalization_1/moments/variance?
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_1/batchnorm/add/y?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"2%
#layer_normalization_1/batchnorm/add?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*"
_output_shapes
:2"2'
%layer_normalization_1/batchnorm/Rsqrt?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization_1/batchnorm/mul?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/mul_1?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/mul_2?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 2%
#layer_normalization_1/batchnorm/sub?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 2'
%layer_normalization_1/batchnorm/add_1?
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*
T0*"
_output_shapes
:2" 2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:2" ::::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:J F
"
_output_shapes
:2" 
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_17516
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_168882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:2o
"
_user_specified_name
inputs/0:LH
"
_output_shapes
:2o
"
_user_specified_name
inputs/1
?	
?
B__inference_dense_3_layer_call_and_return_conditional_losses_16737

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:22	
BiasAddX
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

:22	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*%
_input_shapes
:2 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:2 
 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_18000

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2?2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2?2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	2?:G C

_output_shapes
:	2?
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_16939
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_168882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:2o
!
_user_specified_name	input_1:KG
"
_output_shapes
:2o
!
_user_specified_name	input_2
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_17611

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2o2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*&
_output_shapes
:2f*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*"
_output_shapes
:2f*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2f2	
BiasAddS
ReluReluBiasAdd:output:0*
T0*"
_output_shapes
:2f2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*"
_output_shapes
:2f2

Identity"
identityIdentity:output:0*)
_input_shapes
:2o::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:J F
"
_output_shapes
:2o
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_17664

inputs

identity_1U
IdentityIdentityinputs*
T0*"
_output_shapes
:2"2

Identityd

Identity_1IdentityIdentity:output:0*
T0*"
_output_shapes
:2"2

Identity_1"!

identity_1Identity_1:output:0*!
_input_shapes
:2":J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
?
?
A__inference_conv1d_layer_call_and_return_conditional_losses_16241

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2o2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*&
_output_shapes
:2f*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*"
_output_shapes
:2f*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2f2	
BiasAddS
ReluReluBiasAdd:output:0*
T0*"
_output_shapes
:2f2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*"
_output_shapes
:2f2

Identity"
identityIdentity:output:0*)
_input_shapes
:2o::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:J F
"
_output_shapes
:2o
 
_user_specified_nameinputs
?

a
B__inference_dropout_layer_call_and_return_conditional_losses_16271

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constn
dropout/MulMulinputsdropout/Const:output:0*
T0*"
_output_shapes
:2"2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "      2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*"
_output_shapes
:2"*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2"2
dropout/GreaterEqualz
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2"2
dropout/Castu
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*"
_output_shapes
:2"2
dropout/Mul_1`
IdentityIdentitydropout/Mul_1:z:0*
T0*"
_output_shapes
:2"2

Identity"
identityIdentity:output:0*!
_input_shapes
:2":J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
?

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_16681

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2?2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"2   @  2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	2?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2?2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2?2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2?2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2?2

Identity"
identityIdentity:output:0*
_input_shapes
:	2?:G C

_output_shapes
:	2?
 
_user_specified_nameinputs
?7
?
@__inference_model_layer_call_and_return_conditional_losses_17007

inputs
inputs_1
conv1d_1_16946
conv1d_1_16948
conv1d_16951
conv1d_16953
transformer_block_16961
transformer_block_16963
transformer_block_16965
transformer_block_16967
transformer_block_16969
transformer_block_16971
transformer_block_16973
transformer_block_16975
transformer_block_16977
transformer_block_16979
transformer_block_16981
transformer_block_16983
transformer_block_16985
transformer_block_16987
transformer_block_16989
transformer_block_16991
dense_2_16996
dense_2_16998
dense_3_17001
dense_3_17003
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?)transformer_block/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv1d_1_16946conv1d_1_16948*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_162092"
 conv1d_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_16951conv1d_16953*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_162412 
conv1d/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_161822!
max_pooling1d_1/PartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_161672
max_pooling1d/PartitionedCall?
dropout/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_162762
dropout/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_163062
dropout_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_163262
concatenate/PartitionedCall?
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_block_16961transformer_block_16963transformer_block_16965transformer_block_16967transformer_block_16969transformer_block_16971transformer_block_16973transformer_block_16975transformer_block_16977transformer_block_16979transformer_block_16981transformer_block_16983transformer_block_16985transformer_block_16987transformer_block_16989transformer_block_16991*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2" *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_165462+
)transformer_block/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_166612
flatten/PartitionedCall?
dropout_4/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_166862
dropout_4/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_2_16996dense_2_16998*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_167102!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_17001dense_3_17003*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_167372!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:J F
"
_output_shapes
:2o
 
_user_specified_nameinputs:JF
"
_output_shapes
:2o
 
_user_specified_nameinputs
?
}
(__inference_conv1d_1_layer_call_fn_17620

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_162092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:2f2

Identity"
identityIdentity:output:0*)
_input_shapes
:2o::22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:2o
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_17313
inputs_0
inputs_18
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resourceV
Rtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resourceL
Htransformer_block_multi_head_attention_query_add_readvariableop_resourceT
Ptransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resourceJ
Ftransformer_block_multi_head_attention_key_add_readvariableop_resourceV
Rtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resourceL
Htransformer_block_multi_head_attention_value_add_readvariableop_resourcea
]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resourceW
Stransformer_block_multi_head_attention_attention_output_add_readvariableop_resourceO
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resourceK
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resource=
9transformer_block_dense_tensordot_readvariableop_resource;
7transformer_block_dense_biasadd_readvariableop_resource?
;transformer_block_dense_1_tensordot_readvariableop_resource=
9transformer_block_dense_1_biasadd_readvariableop_resourceQ
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resourceM
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity??conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?.transformer_block/dense/BiasAdd/ReadVariableOp?0transformer_block/dense/Tensordot/ReadVariableOp?0transformer_block/dense_1/BiasAdd/ReadVariableOp?2transformer_block/dense_1/Tensordot/ReadVariableOp?>transformer_block/layer_normalization/batchnorm/ReadVariableOp?Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp?Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp?Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?=transformer_block/multi_head_attention/key/add/ReadVariableOp?Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp??transformer_block/multi_head_attention/query/add/ReadVariableOp?Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp??transformer_block/multi_head_attention/value/add/ReadVariableOp?Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsinputs_1'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2o2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*&
_output_shapes
:2f*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*"
_output_shapes
:2f*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2f2
conv1d_1/BiasAddn
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*"
_output_shapes
:2f2
conv1d_1/Relu?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsinputs_0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2o2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*&
_output_shapes
:2f*
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*"
_output_shapes
:2f*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2f2
conv1d/BiasAddh
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*"
_output_shapes
:2f2
conv1d/Relu?
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim?
max_pooling1d_1/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2f2
max_pooling1d_1/ExpandDims?
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*&
_output_shapes
:2"*
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPool?
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*"
_output_shapes
:2"*
squeeze_dims
2
max_pooling1d_1/Squeeze~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2f2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*&
_output_shapes
:2"*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*"
_output_shapes
:2"*
squeeze_dims
2
max_pooling1d/Squeezes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulmax_pooling1d/Squeeze:output:0dropout/dropout/Const:output:0*
T0*"
_output_shapes
:2"2
dropout/dropout/Mul?
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "      2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*"
_output_shapes
:2"*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2"2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2"2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*"
_output_shapes
:2"2
dropout/dropout/Mul_1w
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul max_pooling1d_1/Squeeze:output:0 dropout_1/dropout/Const:output:0*
T0*"
_output_shapes
:2"2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "      2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*"
_output_shapes
:2"*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2"2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2"2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*"
_output_shapes
:2"2
dropout_1/dropout/Mul_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2dropout/dropout/Mul_1:z:0dropout_1/dropout/Mul_1:z:0 concatenate/concat/axis:output:0*
N*
T0*"
_output_shapes
:2" 2
concatenate/concat?
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02K
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp?
:transformer_block/multi_head_attention/query/einsum/EinsumEinsumconcatenate/concat:output:0Qtransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2<
:transformer_block/multi_head_attention/query/einsum/Einsum?
?transformer_block/multi_head_attention/query/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype02A
?transformer_block/multi_head_attention/query/add/ReadVariableOp?
0transformer_block/multi_head_attention/query/addAddV2Ctransformer_block/multi_head_attention/query/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 22
0transformer_block/multi_head_attention/query/add?
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02I
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp?
8transformer_block/multi_head_attention/key/einsum/EinsumEinsumconcatenate/concat:output:0Otransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2:
8transformer_block/multi_head_attention/key/einsum/Einsum?
=transformer_block/multi_head_attention/key/add/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype02?
=transformer_block/multi_head_attention/key/add/ReadVariableOp?
.transformer_block/multi_head_attention/key/addAddV2Atransformer_block/multi_head_attention/key/einsum/Einsum:output:0Etransformer_block/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 20
.transformer_block/multi_head_attention/key/add?
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02K
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp?
:transformer_block/multi_head_attention/value/einsum/EinsumEinsumconcatenate/concat:output:0Qtransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2<
:transformer_block/multi_head_attention/value/einsum/Einsum?
?transformer_block/multi_head_attention/value/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype02A
?transformer_block/multi_head_attention/value/add/ReadVariableOp?
0transformer_block/multi_head_attention/value/addAddV2Ctransformer_block/multi_head_attention/value/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 22
0transformer_block/multi_head_attention/value/add?
,transformer_block/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2.
,transformer_block/multi_head_attention/Mul/y?
*transformer_block/multi_head_attention/MulMul4transformer_block/multi_head_attention/query/add:z:05transformer_block/multi_head_attention/Mul/y:output:0*
T0*&
_output_shapes
:2" 2,
*transformer_block/multi_head_attention/Mul?
4transformer_block/multi_head_attention/einsum/EinsumEinsum2transformer_block/multi_head_attention/key/add:z:0.transformer_block/multi_head_attention/Mul:z:0*
N*
T0*&
_output_shapes
:2""*
equationaecd,abcd->acbe26
4transformer_block/multi_head_attention/einsum/Einsum?
6transformer_block/multi_head_attention/softmax/SoftmaxSoftmax=transformer_block/multi_head_attention/einsum/Einsum:output:0*
T0*&
_output_shapes
:2""28
6transformer_block/multi_head_attention/softmax/Softmax?
<transformer_block/multi_head_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2>
<transformer_block/multi_head_attention/dropout/dropout/Const?
:transformer_block/multi_head_attention/dropout/dropout/MulMul@transformer_block/multi_head_attention/softmax/Softmax:softmax:0Etransformer_block/multi_head_attention/dropout/dropout/Const:output:0*
T0*&
_output_shapes
:2""2<
:transformer_block/multi_head_attention/dropout/dropout/Mul?
<transformer_block/multi_head_attention/dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"2      "   "   2>
<transformer_block/multi_head_attention/dropout/dropout/Shape?
Stransformer_block/multi_head_attention/dropout/dropout/random_uniform/RandomUniformRandomUniformEtransformer_block/multi_head_attention/dropout/dropout/Shape:output:0*
T0*&
_output_shapes
:2""*
dtype02U
Stransformer_block/multi_head_attention/dropout/dropout/random_uniform/RandomUniform?
Etransformer_block/multi_head_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2G
Etransformer_block/multi_head_attention/dropout/dropout/GreaterEqual/y?
Ctransformer_block/multi_head_attention/dropout/dropout/GreaterEqualGreaterEqual\transformer_block/multi_head_attention/dropout/dropout/random_uniform/RandomUniform:output:0Ntransformer_block/multi_head_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:2""2E
Ctransformer_block/multi_head_attention/dropout/dropout/GreaterEqual?
;transformer_block/multi_head_attention/dropout/dropout/CastCastGtransformer_block/multi_head_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*&
_output_shapes
:2""2=
;transformer_block/multi_head_attention/dropout/dropout/Cast?
<transformer_block/multi_head_attention/dropout/dropout/Mul_1Mul>transformer_block/multi_head_attention/dropout/dropout/Mul:z:0?transformer_block/multi_head_attention/dropout/dropout/Cast:y:0*
T0*&
_output_shapes
:2""2>
<transformer_block/multi_head_attention/dropout/dropout/Mul_1?
6transformer_block/multi_head_attention/einsum_1/EinsumEinsum@transformer_block/multi_head_attention/dropout/dropout/Mul_1:z:04transformer_block/multi_head_attention/value/add:z:0*
N*
T0*&
_output_shapes
:2" *
equationacbe,aecd->abcd28
6transformer_block/multi_head_attention/einsum_1/Einsum?
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02V
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?
Etransformer_block/multi_head_attention/attention_output/einsum/EinsumEinsum?transformer_block/multi_head_attention/einsum_1/Einsum:output:0\transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*"
_output_shapes
:2" *
equationabcd,cde->abe2G
Etransformer_block/multi_head_attention/attention_output/einsum/Einsum?
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02L
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp?
;transformer_block/multi_head_attention/attention_output/addAddV2Ntransformer_block/multi_head_attention/attention_output/einsum/Einsum:output:0Rtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2=
;transformer_block/multi_head_attention/attention_output/add?
)transformer_block/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2+
)transformer_block/dropout_2/dropout/Const?
'transformer_block/dropout_2/dropout/MulMul?transformer_block/multi_head_attention/attention_output/add:z:02transformer_block/dropout_2/dropout/Const:output:0*
T0*"
_output_shapes
:2" 2)
'transformer_block/dropout_2/dropout/Mul?
)transformer_block/dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2+
)transformer_block/dropout_2/dropout/Shape?
@transformer_block/dropout_2/dropout/random_uniform/RandomUniformRandomUniform2transformer_block/dropout_2/dropout/Shape:output:0*
T0*"
_output_shapes
:2" *
dtype02B
@transformer_block/dropout_2/dropout/random_uniform/RandomUniform?
2transformer_block/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=24
2transformer_block/dropout_2/dropout/GreaterEqual/y?
0transformer_block/dropout_2/dropout/GreaterEqualGreaterEqualItransformer_block/dropout_2/dropout/random_uniform/RandomUniform:output:0;transformer_block/dropout_2/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2" 22
0transformer_block/dropout_2/dropout/GreaterEqual?
(transformer_block/dropout_2/dropout/CastCast4transformer_block/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2" 2*
(transformer_block/dropout_2/dropout/Cast?
)transformer_block/dropout_2/dropout/Mul_1Mul+transformer_block/dropout_2/dropout/Mul:z:0,transformer_block/dropout_2/dropout/Cast:y:0*
T0*"
_output_shapes
:2" 2+
)transformer_block/dropout_2/dropout/Mul_1?
transformer_block/addAddV2concatenate/concat:output:0-transformer_block/dropout_2/dropout/Mul_1:z:0*
T0*"
_output_shapes
:2" 2
transformer_block/add?
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2F
Dtransformer_block/layer_normalization/moments/mean/reduction_indices?
2transformer_block/layer_normalization/moments/meanMeantransformer_block/add:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(24
2transformer_block/layer_normalization/moments/mean?
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*
T0*"
_output_shapes
:2"2<
:transformer_block/layer_normalization/moments/StopGradient?
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 2A
?transformer_block/layer_normalization/moments/SquaredDifference?
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block/layer_normalization/moments/variance/reduction_indices?
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(28
6transformer_block/layer_normalization/moments/variance?
5transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?527
5transformer_block/layer_normalization/batchnorm/add/y?
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"25
3transformer_block/layer_normalization/batchnorm/add?
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*"
_output_shapes
:2"27
5transformer_block/layer_normalization/batchnorm/Rsqrt?
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 25
3transformer_block/layer_normalization/batchnorm/mul?
5transformer_block/layer_normalization/batchnorm/mul_1Multransformer_block/add:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 27
5transformer_block/layer_normalization/batchnorm/mul_1?
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 27
5transformer_block/layer_normalization/batchnorm/mul_2?
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02@
>transformer_block/layer_normalization/batchnorm/ReadVariableOp?
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 25
3transformer_block/layer_normalization/batchnorm/sub?
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 27
5transformer_block/layer_normalization/batchnorm/add_1?
0transformer_block/dense/Tensordot/ReadVariableOpReadVariableOp9transformer_block_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype022
0transformer_block/dense/Tensordot/ReadVariableOp?
/transformer_block/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      21
/transformer_block/dense/Tensordot/Reshape/shape?
)transformer_block/dense/Tensordot/ReshapeReshape9transformer_block/layer_normalization/batchnorm/add_1:z:08transformer_block/dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2+
)transformer_block/dense/Tensordot/Reshape?
(transformer_block/dense/Tensordot/MatMulMatMul2transformer_block/dense/Tensordot/Reshape:output:08transformer_block/dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2*
(transformer_block/dense/Tensordot/MatMul?
'transformer_block/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2)
'transformer_block/dense/Tensordot/shape?
!transformer_block/dense/TensordotReshape2transformer_block/dense/Tensordot/MatMul:product:00transformer_block/dense/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2#
!transformer_block/dense/Tensordot?
.transformer_block/dense/BiasAdd/ReadVariableOpReadVariableOp7transformer_block_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.transformer_block/dense/BiasAdd/ReadVariableOp?
transformer_block/dense/BiasAddBiasAdd*transformer_block/dense/Tensordot:output:06transformer_block/dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2!
transformer_block/dense/BiasAdd?
transformer_block/dense/ReluRelu(transformer_block/dense/BiasAdd:output:0*
T0*"
_output_shapes
:2" 2
transformer_block/dense/Relu?
2transformer_block/dense_1/Tensordot/ReadVariableOpReadVariableOp;transformer_block_dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype024
2transformer_block/dense_1/Tensordot/ReadVariableOp?
1transformer_block/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      23
1transformer_block/dense_1/Tensordot/Reshape/shape?
+transformer_block/dense_1/Tensordot/ReshapeReshape*transformer_block/dense/Relu:activations:0:transformer_block/dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 2-
+transformer_block/dense_1/Tensordot/Reshape?
*transformer_block/dense_1/Tensordot/MatMulMatMul4transformer_block/dense_1/Tensordot/Reshape:output:0:transformer_block/dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2,
*transformer_block/dense_1/Tensordot/MatMul?
)transformer_block/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2+
)transformer_block/dense_1/Tensordot/shape?
#transformer_block/dense_1/TensordotReshape4transformer_block/dense_1/Tensordot/MatMul:product:02transformer_block/dense_1/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2%
#transformer_block/dense_1/Tensordot?
0transformer_block/dense_1/BiasAdd/ReadVariableOpReadVariableOp9transformer_block_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0transformer_block/dense_1/BiasAdd/ReadVariableOp?
!transformer_block/dense_1/BiasAddBiasAdd,transformer_block/dense_1/Tensordot:output:08transformer_block/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2#
!transformer_block/dense_1/BiasAdd?
)transformer_block/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2+
)transformer_block/dropout_3/dropout/Const?
'transformer_block/dropout_3/dropout/MulMul*transformer_block/dense_1/BiasAdd:output:02transformer_block/dropout_3/dropout/Const:output:0*
T0*"
_output_shapes
:2" 2)
'transformer_block/dropout_3/dropout/Mul?
)transformer_block/dropout_3/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2+
)transformer_block/dropout_3/dropout/Shape?
@transformer_block/dropout_3/dropout/random_uniform/RandomUniformRandomUniform2transformer_block/dropout_3/dropout/Shape:output:0*
T0*"
_output_shapes
:2" *
dtype02B
@transformer_block/dropout_3/dropout/random_uniform/RandomUniform?
2transformer_block/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=24
2transformer_block/dropout_3/dropout/GreaterEqual/y?
0transformer_block/dropout_3/dropout/GreaterEqualGreaterEqualItransformer_block/dropout_3/dropout/random_uniform/RandomUniform:output:0;transformer_block/dropout_3/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2" 22
0transformer_block/dropout_3/dropout/GreaterEqual?
(transformer_block/dropout_3/dropout/CastCast4transformer_block/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2" 2*
(transformer_block/dropout_3/dropout/Cast?
)transformer_block/dropout_3/dropout/Mul_1Mul+transformer_block/dropout_3/dropout/Mul:z:0,transformer_block/dropout_3/dropout/Cast:y:0*
T0*"
_output_shapes
:2" 2+
)transformer_block/dropout_3/dropout/Mul_1?
transformer_block/add_1AddV29transformer_block/layer_normalization/batchnorm/add_1:z:0-transformer_block/dropout_3/dropout/Mul_1:z:0*
T0*"
_output_shapes
:2" 2
transformer_block/add_1?
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2H
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indices?
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add_1:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(26
4transformer_block/layer_normalization_1/moments/mean?
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:2"2>
<transformer_block/layer_normalization_1/moments/StopGradient?
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add_1:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 2C
Atransformer_block/layer_normalization_1/moments/SquaredDifference?
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indices?
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2:
8transformer_block/layer_normalization_1/moments/variance?
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?529
7transformer_block/layer_normalization_1/batchnorm/add/y?
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"27
5transformer_block/layer_normalization_1/batchnorm/add?
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*"
_output_shapes
:2"29
7transformer_block/layer_normalization_1/batchnorm/Rsqrt?
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02F
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp?
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 27
5transformer_block/layer_normalization_1/batchnorm/mul?
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add_1:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 29
7transformer_block/layer_normalization_1/batchnorm/mul_1?
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 29
7transformer_block/layer_normalization_1/batchnorm/mul_2?
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 27
5transformer_block/layer_normalization_1/batchnorm/sub?
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 29
7transformer_block/layer_normalization_1/batchnorm/add_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten/Const?
flatten/ReshapeReshape;transformer_block/layer_normalization_1/batchnorm/add_1:z:0flatten/Const:output:0*
T0*
_output_shapes
:	2?2
flatten/Reshapew
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_4/dropout/Const?
dropout_4/dropout/MulMulflatten/Reshape:output:0 dropout_4/dropout/Const:output:0*
T0*
_output_shapes
:	2?2
dropout_4/dropout/Mul?
dropout_4/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"2   @  2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*
_output_shapes
:	2?*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2?2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2?2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*
_output_shapes
:	2?2
dropout_4/dropout/Mul_1?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2 2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2 2
dense_2/BiasAddg
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*
_output_shapes

:2 2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:22
dense_3/BiasAddp
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*
_output_shapes

:22
dense_3/Softmax?
IdentityIdentitydense_3/Softmax:softmax:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp/^transformer_block/dense/BiasAdd/ReadVariableOp1^transformer_block/dense/Tensordot/ReadVariableOp1^transformer_block/dense_1/BiasAdd/ReadVariableOp3^transformer_block/dense_1/Tensordot/ReadVariableOp?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpK^transformer_block/multi_head_attention/attention_output/add/ReadVariableOpU^transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp>^transformer_block/multi_head_attention/key/add/ReadVariableOpH^transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/query/add/ReadVariableOpJ^transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/value/add/ReadVariableOpJ^transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2`
.transformer_block/dense/BiasAdd/ReadVariableOp.transformer_block/dense/BiasAdd/ReadVariableOp2d
0transformer_block/dense/Tensordot/ReadVariableOp0transformer_block/dense/Tensordot/ReadVariableOp2d
0transformer_block/dense_1/BiasAdd/ReadVariableOp0transformer_block/dense_1/BiasAdd/ReadVariableOp2h
2transformer_block/dense_1/Tensordot/ReadVariableOp2transformer_block/dense_1/Tensordot/ReadVariableOp2?
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2?
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2?
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2?
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpJtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp2?
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpTtransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2~
=transformer_block/multi_head_attention/key/add/ReadVariableOp=transformer_block/multi_head_attention/key/add/ReadVariableOp2?
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpGtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
?transformer_block/multi_head_attention/query/add/ReadVariableOp?transformer_block/multi_head_attention/query/add/ReadVariableOp2?
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
?transformer_block/multi_head_attention/value/add/ReadVariableOp?transformer_block/multi_head_attention/value/add/ReadVariableOp2?
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:L H
"
_output_shapes
:2o
"
_user_specified_name
inputs/0:LH
"
_output_shapes
:2o
"
_user_specified_name
inputs/1
?
p
F__inference_concatenate_layer_call_and_return_conditional_losses_16326

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisz
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*"
_output_shapes
:2" 2
concat^
IdentityIdentityconcat:output:0*
T0*"
_output_shapes
:2" 2

Identity"
identityIdentity:output:0*/
_input_shapes
:2":2":J F
"
_output_shapes
:2"
 
_user_specified_nameinputs:JF
"
_output_shapes
:2"
 
_user_specified_nameinputs
?
|
'__inference_dense_2_layer_call_fn_18030

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_167102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2 2

Identity"
identityIdentity:output:0*&
_input_shapes
:	2?::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	2?
 
_user_specified_nameinputs
?;
?
@__inference_model_layer_call_and_return_conditional_losses_16888

inputs
inputs_1
conv1d_1_16827
conv1d_1_16829
conv1d_16832
conv1d_16834
transformer_block_16842
transformer_block_16844
transformer_block_16846
transformer_block_16848
transformer_block_16850
transformer_block_16852
transformer_block_16854
transformer_block_16856
transformer_block_16858
transformer_block_16860
transformer_block_16862
transformer_block_16864
transformer_block_16866
transformer_block_16868
transformer_block_16870
transformer_block_16872
dense_2_16877
dense_2_16879
dense_3_16882
dense_3_16884
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?)transformer_block/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv1d_1_16827conv1d_1_16829*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_162092"
 conv1d_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_16832conv1d_16834*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_162412 
conv1d/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_161822!
max_pooling1d_1/PartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_161672
max_pooling1d/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_162712!
dropout/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_163012#
!dropout_1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_163262
concatenate/PartitionedCall?
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_block_16842transformer_block_16844transformer_block_16846transformer_block_16848transformer_block_16850transformer_block_16852transformer_block_16854transformer_block_16856transformer_block_16858transformer_block_16860transformer_block_16862transformer_block_16864transformer_block_16866transformer_block_16868transformer_block_16870transformer_block_16872*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2" *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_164512+
)transformer_block/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_166612
flatten/PartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_166812#
!dropout_4/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_2_16877dense_2_16879*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_167102!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_16882dense_3_16884*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_167372!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:J F
"
_output_shapes
:2o
 
_user_specified_nameinputs:JF
"
_output_shapes
:2o
 
_user_specified_nameinputs
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_16209

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2o2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*&
_output_shapes
:2f*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*"
_output_shapes
:2f*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2f2	
BiasAddS
ReluReluBiasAdd:output:0*
T0*"
_output_shapes
:2f2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*"
_output_shapes
:2f2

Identity"
identityIdentity:output:0*)
_input_shapes
:2o::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:J F
"
_output_shapes
:2o
 
_user_specified_nameinputs
?
?
A__inference_conv1d_layer_call_and_return_conditional_losses_17586

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2o2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*&
_output_shapes
:2f*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*"
_output_shapes
:2f*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2f2	
BiasAddS
ReluReluBiasAdd:output:0*
T0*"
_output_shapes
:2f2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*"
_output_shapes
:2f2

Identity"
identityIdentity:output:0*)
_input_shapes
:2o::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:J F
"
_output_shapes
:2o
 
_user_specified_nameinputs
?

?
1__inference_transformer_block_layer_call_fn_17935

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2" *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_164512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:2" 2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:2" ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:2" 
 
_user_specified_nameinputs
?
E
)__inference_dropout_4_layer_call_fn_18010

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_166862
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2?2

Identity"
identityIdentity:output:0*
_input_shapes
:	2?:G C

_output_shapes
:	2?
 
_user_specified_nameinputs
?|
?
__inference__traced_save_18239
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableopR
Nsavev2_transformer_block_multi_head_attention_query_kernel_read_readvariableopP
Lsavev2_transformer_block_multi_head_attention_query_bias_read_readvariableopP
Lsavev2_transformer_block_multi_head_attention_key_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_attention_key_bias_read_readvariableopR
Nsavev2_transformer_block_multi_head_attention_value_kernel_read_readvariableopP
Lsavev2_transformer_block_multi_head_attention_value_bias_read_readvariableop]
Ysavev2_transformer_block_multi_head_attention_attention_output_kernel_read_readvariableop[
Wsavev2_transformer_block_multi_head_attention_attention_output_bias_read_readvariableop=
9savev2_transformer_block_dense_kernel_read_readvariableop;
7savev2_transformer_block_dense_bias_read_readvariableop?
;savev2_transformer_block_dense_1_kernel_read_readvariableop=
9savev2_transformer_block_dense_1_bias_read_readvariableopJ
Fsavev2_transformer_block_layer_normalization_gamma_read_readvariableopI
Esavev2_transformer_block_layer_normalization_beta_read_readvariableopL
Hsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopK
Gsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_rmsprop_conv1d_kernel_rms_read_readvariableop6
2savev2_rmsprop_conv1d_bias_rms_read_readvariableop:
6savev2_rmsprop_conv1d_1_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv1d_1_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_2_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_3_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_3_bias_rms_read_readvariableop^
Zsavev2_rmsprop_transformer_block_multi_head_attention_query_kernel_rms_read_readvariableop\
Xsavev2_rmsprop_transformer_block_multi_head_attention_query_bias_rms_read_readvariableop\
Xsavev2_rmsprop_transformer_block_multi_head_attention_key_kernel_rms_read_readvariableopZ
Vsavev2_rmsprop_transformer_block_multi_head_attention_key_bias_rms_read_readvariableop^
Zsavev2_rmsprop_transformer_block_multi_head_attention_value_kernel_rms_read_readvariableop\
Xsavev2_rmsprop_transformer_block_multi_head_attention_value_bias_rms_read_readvariableopi
esavev2_rmsprop_transformer_block_multi_head_attention_attention_output_kernel_rms_read_readvariableopg
csavev2_rmsprop_transformer_block_multi_head_attention_attention_output_bias_rms_read_readvariableopI
Esavev2_rmsprop_transformer_block_dense_kernel_rms_read_readvariableopG
Csavev2_rmsprop_transformer_block_dense_bias_rms_read_readvariableopK
Gsavev2_rmsprop_transformer_block_dense_1_kernel_rms_read_readvariableopI
Esavev2_rmsprop_transformer_block_dense_1_bias_rms_read_readvariableopV
Rsavev2_rmsprop_transformer_block_layer_normalization_gamma_rms_read_readvariableopU
Qsavev2_rmsprop_transformer_block_layer_normalization_beta_rms_read_readvariableopX
Tsavev2_rmsprop_transformer_block_layer_normalization_1_gamma_rms_read_readvariableopW
Ssavev2_rmsprop_transformer_block_layer_normalization_1_beta_rms_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableopNsavev2_transformer_block_multi_head_attention_query_kernel_read_readvariableopLsavev2_transformer_block_multi_head_attention_query_bias_read_readvariableopLsavev2_transformer_block_multi_head_attention_key_kernel_read_readvariableopJsavev2_transformer_block_multi_head_attention_key_bias_read_readvariableopNsavev2_transformer_block_multi_head_attention_value_kernel_read_readvariableopLsavev2_transformer_block_multi_head_attention_value_bias_read_readvariableopYsavev2_transformer_block_multi_head_attention_attention_output_kernel_read_readvariableopWsavev2_transformer_block_multi_head_attention_attention_output_bias_read_readvariableop9savev2_transformer_block_dense_kernel_read_readvariableop7savev2_transformer_block_dense_bias_read_readvariableop;savev2_transformer_block_dense_1_kernel_read_readvariableop9savev2_transformer_block_dense_1_bias_read_readvariableopFsavev2_transformer_block_layer_normalization_gamma_read_readvariableopEsavev2_transformer_block_layer_normalization_beta_read_readvariableopHsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopGsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_rmsprop_conv1d_kernel_rms_read_readvariableop2savev2_rmsprop_conv1d_bias_rms_read_readvariableop6savev2_rmsprop_conv1d_1_kernel_rms_read_readvariableop4savev2_rmsprop_conv1d_1_bias_rms_read_readvariableop5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop3savev2_rmsprop_dense_2_bias_rms_read_readvariableop5savev2_rmsprop_dense_3_kernel_rms_read_readvariableop3savev2_rmsprop_dense_3_bias_rms_read_readvariableopZsavev2_rmsprop_transformer_block_multi_head_attention_query_kernel_rms_read_readvariableopXsavev2_rmsprop_transformer_block_multi_head_attention_query_bias_rms_read_readvariableopXsavev2_rmsprop_transformer_block_multi_head_attention_key_kernel_rms_read_readvariableopVsavev2_rmsprop_transformer_block_multi_head_attention_key_bias_rms_read_readvariableopZsavev2_rmsprop_transformer_block_multi_head_attention_value_kernel_rms_read_readvariableopXsavev2_rmsprop_transformer_block_multi_head_attention_value_bias_rms_read_readvariableopesavev2_rmsprop_transformer_block_multi_head_attention_attention_output_kernel_rms_read_readvariableopcsavev2_rmsprop_transformer_block_multi_head_attention_attention_output_bias_rms_read_readvariableopEsavev2_rmsprop_transformer_block_dense_kernel_rms_read_readvariableopCsavev2_rmsprop_transformer_block_dense_bias_rms_read_readvariableopGsavev2_rmsprop_transformer_block_dense_1_kernel_rms_read_readvariableopEsavev2_rmsprop_transformer_block_dense_1_bias_rms_read_readvariableopRsavev2_rmsprop_transformer_block_layer_normalization_gamma_rms_read_readvariableopQsavev2_rmsprop_transformer_block_layer_normalization_beta_rms_read_readvariableopTsavev2_rmsprop_transformer_block_layer_normalization_1_gamma_rms_read_readvariableopSsavev2_rmsprop_transformer_block_layer_normalization_1_beta_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
::
::	? : : :: : : : : :  : :  : :  : :  : :  : :  : : : : : : : :
::
::	? : : ::  : :  : :  : :  : :  : :  : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:
: 

_output_shapes
::($
"
_output_shapes
:
: 

_output_shapes
::%!

_output_shapes
:	? : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :( $
"
_output_shapes
:
: !

_output_shapes
::("$
"
_output_shapes
:
: #

_output_shapes
::%$!

_output_shapes
:	? : %

_output_shapes
: :$& 

_output_shapes

: : '

_output_shapes
::(($
"
_output_shapes
:  :$) 

_output_shapes

: :(*$
"
_output_shapes
:  :$+ 

_output_shapes

: :(,$
"
_output_shapes
:  :$- 

_output_shapes

: :(.$
"
_output_shapes
:  : /

_output_shapes
: :$0 

_output_shapes

:  : 1

_output_shapes
: :$2 

_output_shapes

:  : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :8

_output_shapes
: 
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_16276

inputs

identity_1U
IdentityIdentityinputs*
T0*"
_output_shapes
:2"2

Identityd

Identity_1IdentityIdentity:output:0*
T0*"
_output_shapes
:2"2

Identity_1"!

identity_1Identity_1:output:0*!
_input_shapes
:2":J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_16158
input_1
input_2>
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource2
.model_conv1d_1_biasadd_readvariableop_resource<
8model_conv1d_conv1d_expanddims_1_readvariableop_resource0
,model_conv1d_biasadd_readvariableop_resource\
Xmodel_transformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resourceR
Nmodel_transformer_block_multi_head_attention_query_add_readvariableop_resourceZ
Vmodel_transformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resourceP
Lmodel_transformer_block_multi_head_attention_key_add_readvariableop_resource\
Xmodel_transformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resourceR
Nmodel_transformer_block_multi_head_attention_value_add_readvariableop_resourceg
cmodel_transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource]
Ymodel_transformer_block_multi_head_attention_attention_output_add_readvariableop_resourceU
Qmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resourceQ
Mmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resourceC
?model_transformer_block_dense_tensordot_readvariableop_resourceA
=model_transformer_block_dense_biasadd_readvariableop_resourceE
Amodel_transformer_block_dense_1_tensordot_readvariableop_resourceC
?model_transformer_block_dense_1_biasadd_readvariableop_resourceW
Smodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resourceS
Omodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource0
,model_dense_3_matmul_readvariableop_resource1
-model_dense_3_biasadd_readvariableop_resource
identity??#model/conv1d/BiasAdd/ReadVariableOp?/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp?%model/conv1d_1/BiasAdd/ReadVariableOp?1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?$model/dense_2/BiasAdd/ReadVariableOp?#model/dense_2/MatMul/ReadVariableOp?$model/dense_3/BiasAdd/ReadVariableOp?#model/dense_3/MatMul/ReadVariableOp?4model/transformer_block/dense/BiasAdd/ReadVariableOp?6model/transformer_block/dense/Tensordot/ReadVariableOp?6model/transformer_block/dense_1/BiasAdd/ReadVariableOp?8model/transformer_block/dense_1/Tensordot/ReadVariableOp?Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp?Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp?Pmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp?Zmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Cmodel/transformer_block/multi_head_attention/key/add/ReadVariableOp?Mmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Emodel/transformer_block/multi_head_attention/query/add/ReadVariableOp?Omodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Emodel/transformer_block/multi_head_attention/value/add/ReadVariableOp?Omodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp?
$model/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/conv1d_1/conv1d/ExpandDims/dim?
 model/conv1d_1/conv1d/ExpandDims
ExpandDimsinput_2-model/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2o2"
 model/conv1d_1/conv1d/ExpandDims?
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype023
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
&model/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/conv1d_1/conv1d/ExpandDims_1/dim?
"model/conv1d_1/conv1d/ExpandDims_1
ExpandDims9model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2$
"model/conv1d_1/conv1d/ExpandDims_1?
model/conv1d_1/conv1dConv2D)model/conv1d_1/conv1d/ExpandDims:output:0+model/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*&
_output_shapes
:2f*
paddingVALID*
strides
2
model/conv1d_1/conv1d?
model/conv1d_1/conv1d/SqueezeSqueezemodel/conv1d_1/conv1d:output:0*
T0*"
_output_shapes
:2f*
squeeze_dims

?????????2
model/conv1d_1/conv1d/Squeeze?
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv1d_1/BiasAdd/ReadVariableOp?
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/conv1d/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2f2
model/conv1d_1/BiasAdd?
model/conv1d_1/ReluRelumodel/conv1d_1/BiasAdd:output:0*
T0*"
_output_shapes
:2f2
model/conv1d_1/Relu?
"model/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model/conv1d/conv1d/ExpandDims/dim?
model/conv1d/conv1d/ExpandDims
ExpandDimsinput_1+model/conv1d/conv1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2o2 
model/conv1d/conv1d/ExpandDims?
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype021
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
$model/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model/conv1d/conv1d/ExpandDims_1/dim?
 model/conv1d/conv1d/ExpandDims_1
ExpandDims7model/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2"
 model/conv1d/conv1d/ExpandDims_1?
model/conv1d/conv1dConv2D'model/conv1d/conv1d/ExpandDims:output:0)model/conv1d/conv1d/ExpandDims_1:output:0*
T0*&
_output_shapes
:2f*
paddingVALID*
strides
2
model/conv1d/conv1d?
model/conv1d/conv1d/SqueezeSqueezemodel/conv1d/conv1d:output:0*
T0*"
_output_shapes
:2f*
squeeze_dims

?????????2
model/conv1d/conv1d/Squeeze?
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv1d/BiasAdd/ReadVariableOp?
model/conv1d/BiasAddBiasAdd$model/conv1d/conv1d/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2f2
model/conv1d/BiasAddz
model/conv1d/ReluRelumodel/conv1d/BiasAdd:output:0*
T0*"
_output_shapes
:2f2
model/conv1d/Relu?
$model/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/max_pooling1d_1/ExpandDims/dim?
 model/max_pooling1d_1/ExpandDims
ExpandDims!model/conv1d_1/Relu:activations:0-model/max_pooling1d_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2f2"
 model/max_pooling1d_1/ExpandDims?
model/max_pooling1d_1/MaxPoolMaxPool)model/max_pooling1d_1/ExpandDims:output:0*&
_output_shapes
:2"*
ksize
*
paddingVALID*
strides
2
model/max_pooling1d_1/MaxPool?
model/max_pooling1d_1/SqueezeSqueeze&model/max_pooling1d_1/MaxPool:output:0*
T0*"
_output_shapes
:2"*
squeeze_dims
2
model/max_pooling1d_1/Squeeze?
"model/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/max_pooling1d/ExpandDims/dim?
model/max_pooling1d/ExpandDims
ExpandDimsmodel/conv1d/Relu:activations:0+model/max_pooling1d/ExpandDims/dim:output:0*
T0*&
_output_shapes
:2f2 
model/max_pooling1d/ExpandDims?
model/max_pooling1d/MaxPoolMaxPool'model/max_pooling1d/ExpandDims:output:0*&
_output_shapes
:2"*
ksize
*
paddingVALID*
strides
2
model/max_pooling1d/MaxPool?
model/max_pooling1d/SqueezeSqueeze$model/max_pooling1d/MaxPool:output:0*
T0*"
_output_shapes
:2"*
squeeze_dims
2
model/max_pooling1d/Squeeze?
model/dropout/IdentityIdentity$model/max_pooling1d/Squeeze:output:0*
T0*"
_output_shapes
:2"2
model/dropout/Identity?
model/dropout_1/IdentityIdentity&model/max_pooling1d_1/Squeeze:output:0*
T0*"
_output_shapes
:2"2
model/dropout_1/Identity?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2model/dropout/Identity:output:0!model/dropout_1/Identity:output:0&model/concatenate/concat/axis:output:0*
N*
T0*"
_output_shapes
:2" 2
model/concatenate/concat?
Omodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Q
Omodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp?
@model/transformer_block/multi_head_attention/query/einsum/EinsumEinsum!model/concatenate/concat:output:0Wmodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2B
@model/transformer_block/multi_head_attention/query/einsum/Einsum?
Emodel/transformer_block/multi_head_attention/query/add/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype02G
Emodel/transformer_block/multi_head_attention/query/add/ReadVariableOp?
6model/transformer_block/multi_head_attention/query/addAddV2Imodel/transformer_block/multi_head_attention/query/einsum/Einsum:output:0Mmodel/transformer_block/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 28
6model/transformer_block/multi_head_attention/query/add?
Mmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpVmodel_transformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp?
>model/transformer_block/multi_head_attention/key/einsum/EinsumEinsum!model/concatenate/concat:output:0Umodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2@
>model/transformer_block/multi_head_attention/key/einsum/Einsum?
Cmodel/transformer_block/multi_head_attention/key/add/ReadVariableOpReadVariableOpLmodel_transformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Cmodel/transformer_block/multi_head_attention/key/add/ReadVariableOp?
4model/transformer_block/multi_head_attention/key/addAddV2Gmodel/transformer_block/multi_head_attention/key/einsum/Einsum:output:0Kmodel/transformer_block/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 26
4model/transformer_block/multi_head_attention/key/add?
Omodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Q
Omodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp?
@model/transformer_block/multi_head_attention/value/einsum/EinsumEinsum!model/concatenate/concat:output:0Wmodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*&
_output_shapes
:2" *
equationabc,cde->abde2B
@model/transformer_block/multi_head_attention/value/einsum/Einsum?
Emodel/transformer_block/multi_head_attention/value/add/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype02G
Emodel/transformer_block/multi_head_attention/value/add/ReadVariableOp?
6model/transformer_block/multi_head_attention/value/addAddV2Imodel/transformer_block/multi_head_attention/value/einsum/Einsum:output:0Mmodel/transformer_block/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*&
_output_shapes
:2" 28
6model/transformer_block/multi_head_attention/value/add?
2model/transformer_block/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>24
2model/transformer_block/multi_head_attention/Mul/y?
0model/transformer_block/multi_head_attention/MulMul:model/transformer_block/multi_head_attention/query/add:z:0;model/transformer_block/multi_head_attention/Mul/y:output:0*
T0*&
_output_shapes
:2" 22
0model/transformer_block/multi_head_attention/Mul?
:model/transformer_block/multi_head_attention/einsum/EinsumEinsum8model/transformer_block/multi_head_attention/key/add:z:04model/transformer_block/multi_head_attention/Mul:z:0*
N*
T0*&
_output_shapes
:2""*
equationaecd,abcd->acbe2<
:model/transformer_block/multi_head_attention/einsum/Einsum?
<model/transformer_block/multi_head_attention/softmax/SoftmaxSoftmaxCmodel/transformer_block/multi_head_attention/einsum/Einsum:output:0*
T0*&
_output_shapes
:2""2>
<model/transformer_block/multi_head_attention/softmax/Softmax?
=model/transformer_block/multi_head_attention/dropout/IdentityIdentityFmodel/transformer_block/multi_head_attention/softmax/Softmax:softmax:0*
T0*&
_output_shapes
:2""2?
=model/transformer_block/multi_head_attention/dropout/Identity?
<model/transformer_block/multi_head_attention/einsum_1/EinsumEinsumFmodel/transformer_block/multi_head_attention/dropout/Identity:output:0:model/transformer_block/multi_head_attention/value/add:z:0*
N*
T0*&
_output_shapes
:2" *
equationacbe,aecd->abcd2>
<model/transformer_block/multi_head_attention/einsum_1/Einsum?
Zmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpcmodel_transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02\
Zmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?
Kmodel/transformer_block/multi_head_attention/attention_output/einsum/EinsumEinsumEmodel/transformer_block/multi_head_attention/einsum_1/Einsum:output:0bmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*"
_output_shapes
:2" *
equationabcd,cde->abe2M
Kmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum?
Pmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpYmodel_transformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02R
Pmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp?
Amodel/transformer_block/multi_head_attention/attention_output/addAddV2Tmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum:output:0Xmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2C
Amodel/transformer_block/multi_head_attention/attention_output/add?
*model/transformer_block/dropout_2/IdentityIdentityEmodel/transformer_block/multi_head_attention/attention_output/add:z:0*
T0*"
_output_shapes
:2" 2,
*model/transformer_block/dropout_2/Identity?
model/transformer_block/addAddV2!model/concatenate/concat:output:03model/transformer_block/dropout_2/Identity:output:0*
T0*"
_output_shapes
:2" 2
model/transformer_block/add?
Jmodel/transformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jmodel/transformer_block/layer_normalization/moments/mean/reduction_indices?
8model/transformer_block/layer_normalization/moments/meanMeanmodel/transformer_block/add:z:0Smodel/transformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2:
8model/transformer_block/layer_normalization/moments/mean?
@model/transformer_block/layer_normalization/moments/StopGradientStopGradientAmodel/transformer_block/layer_normalization/moments/mean:output:0*
T0*"
_output_shapes
:2"2B
@model/transformer_block/layer_normalization/moments/StopGradient?
Emodel/transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencemodel/transformer_block/add:z:0Imodel/transformer_block/layer_normalization/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 2G
Emodel/transformer_block/layer_normalization/moments/SquaredDifference?
Nmodel/transformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Nmodel/transformer_block/layer_normalization/moments/variance/reduction_indices?
<model/transformer_block/layer_normalization/moments/varianceMeanImodel/transformer_block/layer_normalization/moments/SquaredDifference:z:0Wmodel/transformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2>
<model/transformer_block/layer_normalization/moments/variance?
;model/transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52=
;model/transformer_block/layer_normalization/batchnorm/add/y?
9model/transformer_block/layer_normalization/batchnorm/addAddV2Emodel/transformer_block/layer_normalization/moments/variance:output:0Dmodel/transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"2;
9model/transformer_block/layer_normalization/batchnorm/add?
;model/transformer_block/layer_normalization/batchnorm/RsqrtRsqrt=model/transformer_block/layer_normalization/batchnorm/add:z:0*
T0*"
_output_shapes
:2"2=
;model/transformer_block/layer_normalization/batchnorm/Rsqrt?
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpQmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?
9model/transformer_block/layer_normalization/batchnorm/mulMul?model/transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Pmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2;
9model/transformer_block/layer_normalization/batchnorm/mul?
;model/transformer_block/layer_normalization/batchnorm/mul_1Mulmodel/transformer_block/add:z:0=model/transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2=
;model/transformer_block/layer_normalization/batchnorm/mul_1?
;model/transformer_block/layer_normalization/batchnorm/mul_2MulAmodel/transformer_block/layer_normalization/moments/mean:output:0=model/transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2=
;model/transformer_block/layer_normalization/batchnorm/mul_2?
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpMmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02F
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp?
9model/transformer_block/layer_normalization/batchnorm/subSubLmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp:value:0?model/transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 2;
9model/transformer_block/layer_normalization/batchnorm/sub?
;model/transformer_block/layer_normalization/batchnorm/add_1AddV2?model/transformer_block/layer_normalization/batchnorm/mul_1:z:0=model/transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 2=
;model/transformer_block/layer_normalization/batchnorm/add_1?
6model/transformer_block/dense/Tensordot/ReadVariableOpReadVariableOp?model_transformer_block_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype028
6model/transformer_block/dense/Tensordot/ReadVariableOp?
5model/transformer_block/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      27
5model/transformer_block/dense/Tensordot/Reshape/shape?
/model/transformer_block/dense/Tensordot/ReshapeReshape?model/transformer_block/layer_normalization/batchnorm/add_1:z:0>model/transformer_block/dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 21
/model/transformer_block/dense/Tensordot/Reshape?
.model/transformer_block/dense/Tensordot/MatMulMatMul8model/transformer_block/dense/Tensordot/Reshape:output:0>model/transformer_block/dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 20
.model/transformer_block/dense/Tensordot/MatMul?
-model/transformer_block/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       2/
-model/transformer_block/dense/Tensordot/shape?
'model/transformer_block/dense/TensordotReshape8model/transformer_block/dense/Tensordot/MatMul:product:06model/transformer_block/dense/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2)
'model/transformer_block/dense/Tensordot?
4model/transformer_block/dense/BiasAdd/ReadVariableOpReadVariableOp=model_transformer_block_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4model/transformer_block/dense/BiasAdd/ReadVariableOp?
%model/transformer_block/dense/BiasAddBiasAdd0model/transformer_block/dense/Tensordot:output:0<model/transformer_block/dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2'
%model/transformer_block/dense/BiasAdd?
"model/transformer_block/dense/ReluRelu.model/transformer_block/dense/BiasAdd:output:0*
T0*"
_output_shapes
:2" 2$
"model/transformer_block/dense/Relu?
8model/transformer_block/dense_1/Tensordot/ReadVariableOpReadVariableOpAmodel_transformer_block_dense_1_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02:
8model/transformer_block/dense_1/Tensordot/ReadVariableOp?
7model/transformer_block/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      29
7model/transformer_block/dense_1/Tensordot/Reshape/shape?
1model/transformer_block/dense_1/Tensordot/ReshapeReshape0model/transformer_block/dense/Relu:activations:0@model/transformer_block/dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	? 23
1model/transformer_block/dense_1/Tensordot/Reshape?
0model/transformer_block/dense_1/Tensordot/MatMulMatMul:model/transformer_block/dense_1/Tensordot/Reshape:output:0@model/transformer_block/dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 22
0model/transformer_block/dense_1/Tensordot/MatMul?
/model/transformer_block/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "       21
/model/transformer_block/dense_1/Tensordot/shape?
)model/transformer_block/dense_1/TensordotReshape:model/transformer_block/dense_1/Tensordot/MatMul:product:08model/transformer_block/dense_1/Tensordot/shape:output:0*
T0*"
_output_shapes
:2" 2+
)model/transformer_block/dense_1/Tensordot?
6model/transformer_block/dense_1/BiasAdd/ReadVariableOpReadVariableOp?model_transformer_block_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6model/transformer_block/dense_1/BiasAdd/ReadVariableOp?
'model/transformer_block/dense_1/BiasAddBiasAdd2model/transformer_block/dense_1/Tensordot:output:0>model/transformer_block/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2)
'model/transformer_block/dense_1/BiasAdd?
*model/transformer_block/dropout_3/IdentityIdentity0model/transformer_block/dense_1/BiasAdd:output:0*
T0*"
_output_shapes
:2" 2,
*model/transformer_block/dropout_3/Identity?
model/transformer_block/add_1AddV2?model/transformer_block/layer_normalization/batchnorm/add_1:z:03model/transformer_block/dropout_3/Identity:output:0*
T0*"
_output_shapes
:2" 2
model/transformer_block/add_1?
Lmodel/transformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Lmodel/transformer_block/layer_normalization_1/moments/mean/reduction_indices?
:model/transformer_block/layer_normalization_1/moments/meanMean!model/transformer_block/add_1:z:0Umodel/transformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2<
:model/transformer_block/layer_normalization_1/moments/mean?
Bmodel/transformer_block/layer_normalization_1/moments/StopGradientStopGradientCmodel/transformer_block/layer_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:2"2D
Bmodel/transformer_block/layer_normalization_1/moments/StopGradient?
Gmodel/transformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifference!model/transformer_block/add_1:z:0Kmodel/transformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*"
_output_shapes
:2" 2I
Gmodel/transformer_block/layer_normalization_1/moments/SquaredDifference?
Pmodel/transformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel/transformer_block/layer_normalization_1/moments/variance/reduction_indices?
>model/transformer_block/layer_normalization_1/moments/varianceMeanKmodel/transformer_block/layer_normalization_1/moments/SquaredDifference:z:0Ymodel/transformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:2"*
	keep_dims(2@
>model/transformer_block/layer_normalization_1/moments/variance?
=model/transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52?
=model/transformer_block/layer_normalization_1/batchnorm/add/y?
;model/transformer_block/layer_normalization_1/batchnorm/addAddV2Gmodel/transformer_block/layer_normalization_1/moments/variance:output:0Fmodel/transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*"
_output_shapes
:2"2=
;model/transformer_block/layer_normalization_1/batchnorm/add?
=model/transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt?model/transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*"
_output_shapes
:2"2?
=model/transformer_block/layer_normalization_1/batchnorm/Rsqrt?
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpSmodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp?
;model/transformer_block/layer_normalization_1/batchnorm/mulMulAmodel/transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Rmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:2" 2=
;model/transformer_block/layer_normalization_1/batchnorm/mul?
=model/transformer_block/layer_normalization_1/batchnorm/mul_1Mul!model/transformer_block/add_1:z:0?model/transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2?
=model/transformer_block/layer_normalization_1/batchnorm/mul_1?
=model/transformer_block/layer_normalization_1/batchnorm/mul_2MulCmodel/transformer_block/layer_normalization_1/moments/mean:output:0?model/transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*"
_output_shapes
:2" 2?
=model/transformer_block/layer_normalization_1/batchnorm/mul_2?
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpOmodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02H
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?
;model/transformer_block/layer_normalization_1/batchnorm/subSubNmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0Amodel/transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:2" 2=
;model/transformer_block/layer_normalization_1/batchnorm/sub?
=model/transformer_block/layer_normalization_1/batchnorm/add_1AddV2Amodel/transformer_block/layer_normalization_1/batchnorm/mul_1:z:0?model/transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*"
_output_shapes
:2" 2?
=model/transformer_block/layer_normalization_1/batchnorm/add_1{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
model/flatten/Const?
model/flatten/ReshapeReshapeAmodel/transformer_block/layer_normalization_1/batchnorm/add_1:z:0model/flatten/Const:output:0*
T0*
_output_shapes
:	2?2
model/flatten/Reshape?
model/dropout_4/IdentityIdentitymodel/flatten/Reshape:output:0*
T0*
_output_shapes
:	2?2
model/dropout_4/Identity?
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02%
#model/dense_2/MatMul/ReadVariableOp?
model/dense_2/MatMulMatMul!model/dropout_4/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2 2
model/dense_2/MatMul?
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2 2
model/dense_2/BiasAddy
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*
_output_shapes

:2 2
model/dense_2/Relu?
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_3/MatMul/ReadVariableOp?
model/dense_3/MatMulMatMul model/dense_2/Relu:activations:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
model/dense_3/MatMul?
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_3/BiasAdd/ReadVariableOp?
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:22
model/dense_3/BiasAdd?
model/dense_3/SoftmaxSoftmaxmodel/dense_3/BiasAdd:output:0*
T0*
_output_shapes

:22
model/dense_3/Softmax?
IdentityIdentitymodel/dense_3/Softmax:softmax:0$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/conv1d/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp5^model/transformer_block/dense/BiasAdd/ReadVariableOp7^model/transformer_block/dense/Tensordot/ReadVariableOp7^model/transformer_block/dense_1/BiasAdd/ReadVariableOp9^model/transformer_block/dense_1/Tensordot/ReadVariableOpE^model/transformer_block/layer_normalization/batchnorm/ReadVariableOpI^model/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpG^model/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpK^model/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpQ^model/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp[^model/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpD^model/transformer_block/multi_head_attention/key/add/ReadVariableOpN^model/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpF^model/transformer_block/multi_head_attention/query/add/ReadVariableOpP^model/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpF^model/transformer_block/multi_head_attention/value/add/ReadVariableOpP^model/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::2J
#model/conv1d/BiasAdd/ReadVariableOp#model/conv1d/BiasAdd/ReadVariableOp2b
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp2N
%model/conv1d_1/BiasAdd/ReadVariableOp%model/conv1d_1/BiasAdd/ReadVariableOp2f
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2l
4model/transformer_block/dense/BiasAdd/ReadVariableOp4model/transformer_block/dense/BiasAdd/ReadVariableOp2p
6model/transformer_block/dense/Tensordot/ReadVariableOp6model/transformer_block/dense/Tensordot/ReadVariableOp2p
6model/transformer_block/dense_1/BiasAdd/ReadVariableOp6model/transformer_block/dense_1/BiasAdd/ReadVariableOp2t
8model/transformer_block/dense_1/Tensordot/ReadVariableOp8model/transformer_block/dense_1/Tensordot/ReadVariableOp2?
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOpDmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp2?
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpHmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2?
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpFmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2?
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpJmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Pmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOpPmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp2?
Zmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpZmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Cmodel/transformer_block/multi_head_attention/key/add/ReadVariableOpCmodel/transformer_block/multi_head_attention/key/add/ReadVariableOp2?
Mmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpMmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Emodel/transformer_block/multi_head_attention/query/add/ReadVariableOpEmodel/transformer_block/multi_head_attention/query/add/ReadVariableOp2?
Omodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpOmodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Emodel/transformer_block/multi_head_attention/value/add/ReadVariableOpEmodel/transformer_block/multi_head_attention/value/add/ReadVariableOp2?
Omodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpOmodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:K G
"
_output_shapes
:2o
!
_user_specified_name	input_1:KG
"
_output_shapes
:2o
!
_user_specified_name	input_2
?

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_16301

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constn
dropout/MulMulinputsdropout/Const:output:0*
T0*"
_output_shapes
:2"2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "      2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*"
_output_shapes
:2"*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2"2
dropout/GreaterEqualz
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2"2
dropout/Castu
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*"
_output_shapes
:2"2
dropout/Mul_1`
IdentityIdentitydropout/Mul_1:z:0*
T0*"
_output_shapes
:2"2

Identity"
identityIdentity:output:0*!
_input_shapes
:2":J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_16661

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Const_
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	2?2	
Reshape\
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	2?2

Identity"
identityIdentity:output:0*!
_input_shapes
:2" :J F
"
_output_shapes
:2" 
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_17983

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_166612
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2?2

Identity"
identityIdentity:output:0*!
_input_shapes
:2" :J F
"
_output_shapes
:2" 
 
_user_specified_nameinputs
?;
?
@__inference_model_layer_call_and_return_conditional_losses_16754
input_1
input_2
conv1d_1_16220
conv1d_1_16222
conv1d_16252
conv1d_16254
transformer_block_16622
transformer_block_16624
transformer_block_16626
transformer_block_16628
transformer_block_16630
transformer_block_16632
transformer_block_16634
transformer_block_16636
transformer_block_16638
transformer_block_16640
transformer_block_16642
transformer_block_16644
transformer_block_16646
transformer_block_16648
transformer_block_16650
transformer_block_16652
dense_2_16721
dense_2_16723
dense_3_16748
dense_3_16750
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?)transformer_block/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv1d_1_16220conv1d_1_16222*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_162092"
 conv1d_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_16252conv1d_16254*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_162412 
conv1d/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_161822!
max_pooling1d_1/PartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_161672
max_pooling1d/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_162712!
dropout/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_163012#
!dropout_1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_163262
concatenate/PartitionedCall?
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_block_16622transformer_block_16624transformer_block_16626transformer_block_16628transformer_block_16630transformer_block_16632transformer_block_16634transformer_block_16636transformer_block_16638transformer_block_16640transformer_block_16642transformer_block_16644transformer_block_16646transformer_block_16648transformer_block_16650transformer_block_16652*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2" *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_164512+
)transformer_block/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_166612
flatten/PartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_166812#
!dropout_4/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_2_16721dense_2_16723*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_167102!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_16748dense_3_16750*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_167372!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:K G
"
_output_shapes
:2o
!
_user_specified_name	input_1:KG
"
_output_shapes
:2o
!
_user_specified_name	input_2
?
r
F__inference_concatenate_layer_call_and_return_conditional_losses_17681
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis|
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*"
_output_shapes
:2" 2
concat^
IdentityIdentityconcat:output:0*
T0*"
_output_shapes
:2" 2

Identity"
identityIdentity:output:0*/
_input_shapes
:2":2":L H
"
_output_shapes
:2"
"
_user_specified_name
inputs/0:LH
"
_output_shapes
:2"
"
_user_specified_name
inputs/1
?

?
1__inference_transformer_block_layer_call_fn_17972

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2" *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_165462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:2" 2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:2" ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:2" 
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_17570
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_170072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:2o
"
_user_specified_name
inputs/0:LH
"
_output_shapes
:2o
"
_user_specified_name
inputs/1
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_17637

inputs

identity_1U
IdentityIdentityinputs*
T0*"
_output_shapes
:2"2

Identityd

Identity_1IdentityIdentity:output:0*
T0*"
_output_shapes
:2"2

Identity_1"!

identity_1Identity_1:output:0*!
_input_shapes
:2":J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_18021

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2 2	
BiasAddO
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:2 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2 2

Identity"
identityIdentity:output:0*&
_input_shapes
:	2?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	2?
 
_user_specified_nameinputs
?

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_17659

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constn
dropout/MulMulinputsdropout/Const:output:0*
T0*"
_output_shapes
:2"2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2   "      2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*"
_output_shapes
:2"*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:2"2
dropout/GreaterEqualz
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2"2
dropout/Castu
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*"
_output_shapes
:2"2
dropout/Mul_1`
IdentityIdentitydropout/Mul_1:z:0*
T0*"
_output_shapes
:2"2

Identity"
identityIdentity:output:0*!
_input_shapes
:2":J F
"
_output_shapes
:2"
 
_user_specified_nameinputs
?7
?
@__inference_model_layer_call_and_return_conditional_losses_16819
input_1
input_2
conv1d_1_16758
conv1d_1_16760
conv1d_16763
conv1d_16765
transformer_block_16773
transformer_block_16775
transformer_block_16777
transformer_block_16779
transformer_block_16781
transformer_block_16783
transformer_block_16785
transformer_block_16787
transformer_block_16789
transformer_block_16791
transformer_block_16793
transformer_block_16795
transformer_block_16797
transformer_block_16799
transformer_block_16801
transformer_block_16803
dense_2_16808
dense_2_16810
dense_3_16813
dense_3_16815
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?)transformer_block/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv1d_1_16758conv1d_1_16760*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_162092"
 conv1d_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_16763conv1d_16765*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2f*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_162412 
conv1d/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_161822!
max_pooling1d_1/PartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_161672
max_pooling1d/PartitionedCall?
dropout/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_162762
dropout/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_163062
dropout_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_163262
concatenate/PartitionedCall?
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_block_16773transformer_block_16775transformer_block_16777transformer_block_16779transformer_block_16781transformer_block_16783transformer_block_16785transformer_block_16787transformer_block_16789transformer_block_16791transformer_block_16793transformer_block_16795transformer_block_16797transformer_block_16799transformer_block_16801transformer_block_16803*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:2" *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_165462+
)transformer_block/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_166612
flatten/PartitionedCall?
dropout_4/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_166862
dropout_4/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_2_16808dense_2_16810*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_167102!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_16813dense_3_16815*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_167372!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*?
_input_shapes~
|:2o:2o::::::::::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:K G
"
_output_shapes
:2o
!
_user_specified_name	input_1:KG
"
_output_shapes
:2o
!
_user_specified_name	input_2
?	
?
B__inference_dense_3_layer_call_and_return_conditional_losses_18041

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:22	
BiasAddX
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

:22	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:22

Identity"
identityIdentity:output:0*%
_input_shapes
:2 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:2 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
6
input_1+
serving_default_input_1:02o
6
input_2+
serving_default_input_2:02o2
dense_3'
StatefulPartitionedCall:02tensorflow/serving/predict:݉
?3
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
layer_with_weights-4
layer-13
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?/
_tf_keras_network?/{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [50, 111, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [50, 111, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [10]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [10]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_1", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling1d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling1d_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["dropout", 0, 0, {}], ["dropout_1", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["transformer_block", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_3", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 111, 4]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 111, 6]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [50, 111, 4]}, {"class_name": "TensorShape", "items": [50, 111, 6]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "categorical_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [50, 111, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [50, 111, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [50, 111, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [50, 111, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [10]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 111, 4]}}
?	

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [10]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 111, 6]}}
?
!trainable_variables
"regularization_losses
#	variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
%trainable_variables
&regularization_losses
'	variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
)trainable_variables
*regularization_losses
+	variables
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
-trainable_variables
.regularization_losses
/	variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
1trainable_variables
2regularization_losses
3	variables
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [50, 34, 16]}, {"class_name": "TensorShape", "items": [50, 34, 16]}]}
?
5att
6ffn1
7ffn2
8
layernorm1
9
layernorm2
:dropout1
;dropout2
<trainable_variables
=regularization_losses
>	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TransformerBlock", "name": "transformer_block", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

Hkernel
Ibias
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1088}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 1088]}}
?

Nkernel
Obias
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 32]}}
?
Titer
	Udecay
Vlearning_rate
Wmomentum
Xrho
rms?
rms?
rms?
rms?
Hrms?
Irms?
Nrms?
Orms?
Yrms?
Zrms?
[rms?
\rms?
]rms?
^rms?
_rms?
`rms?
arms?
brms?
crms?
drms?
erms?
frms?
grms?
hrms?"
	optimizer
?
0
1
2
3
Y4
Z5
[6
\7
]8
^9
_10
`11
a12
b13
c14
d15
e16
f17
g18
h19
H20
I21
N22
O23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
Y4
Z5
[6
\7
]8
^9
_10
`11
a12
b13
c14
d15
e16
f17
g18
h19
H20
I21
N22
O23"
trackable_list_wrapper
?
trainable_variables
regularization_losses
	variables
ilayer_metrics
jlayer_regularization_losses
kmetrics

llayers
mnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
#:!
2conv1d/kernel
:2conv1d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
regularization_losses
	variables
nlayer_metrics
olayer_regularization_losses
pmetrics

qlayers
rnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#
2conv1d_1/kernel
:2conv1d_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
regularization_losses
	variables
slayer_metrics
tlayer_regularization_losses
umetrics

vlayers
wnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
!trainable_variables
"regularization_losses
#	variables
xlayer_metrics
ylayer_regularization_losses
zmetrics

{layers
|non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
%trainable_variables
&regularization_losses
'	variables
}layer_metrics
~layer_regularization_losses
metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
)trainable_variables
*regularization_losses
+	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-trainable_variables
.regularization_losses
/	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
1trainable_variables
2regularization_losses
3	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?_query_dense
?
_key_dense
?_value_dense
?_softmax
?_dropout_layer
?_output_dense
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MultiHeadAttention", "name": "multi_head_attention", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 2, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
?

akernel
bbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 34, 32]}}
?

ckernel
dbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 34, 32]}}
?
	?axis
	egamma
fbeta
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 34, 32]}}
?
	?axis
	ggamma
hbeta
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 34, 32]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15"
trackable_list_wrapper
?
<trainable_variables
=regularization_losses
>	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@trainable_variables
Aregularization_losses
B	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dtrainable_variables
Eregularization_losses
F	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	? 2dense_2/kernel
: 2dense_2/bias
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
Jtrainable_variables
Kregularization_losses
L	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_3/kernel
:2dense_3/bias
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
?
Ptrainable_variables
Qregularization_losses
R	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
I:G  23transformer_block/multi_head_attention/query/kernel
C:A 21transformer_block/multi_head_attention/query/bias
G:E  21transformer_block/multi_head_attention/key/kernel
A:? 2/transformer_block/multi_head_attention/key/bias
I:G  23transformer_block/multi_head_attention/value/kernel
C:A 21transformer_block/multi_head_attention/value/bias
T:R  2>transformer_block/multi_head_attention/attention_output/kernel
J:H 2<transformer_block/multi_head_attention/attention_output/bias
0:.  2transformer_block/dense/kernel
*:( 2transformer_block/dense/bias
2:0  2 transformer_block/dense_1/kernel
,:* 2transformer_block/dense_1/bias
9:7 2+transformer_block/layer_normalization/gamma
8:6 2*transformer_block/layer_normalization/beta
;:9 2-transformer_block/layer_normalization_1/gamma
::8 2,transformer_block/layer_normalization_1/beta
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?partial_output_shape
?full_output_shape

Ykernel
Zbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 2, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 34, 32]}}
?
?partial_output_shape
?full_output_shape

[kernel
\bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 2, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 34, 32]}}
?
?partial_output_shape
?full_output_shape

]kernel
^bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 2, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 34, 32]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
?
?partial_output_shape
?full_output_shape

_kernel
`bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 34, 2, 32]}}
X
Y0
Z1
[2
\3
]4
^5
_6
`7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
Y0
Z1
[2
\3
]4
^5
_6
`7"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Q
50
61
72
83
94
:5
;6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?	variables
?layer_metrics
 ?layer_regularization_losses
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
-:+
2RMSprop/conv1d/kernel/rms
#:!2RMSprop/conv1d/bias/rms
/:-
2RMSprop/conv1d_1/kernel/rms
%:#2RMSprop/conv1d_1/bias/rms
+:)	? 2RMSprop/dense_2/kernel/rms
$:" 2RMSprop/dense_2/bias/rms
*:( 2RMSprop/dense_3/kernel/rms
$:"2RMSprop/dense_3/bias/rms
S:Q  2?RMSprop/transformer_block/multi_head_attention/query/kernel/rms
M:K 2=RMSprop/transformer_block/multi_head_attention/query/bias/rms
Q:O  2=RMSprop/transformer_block/multi_head_attention/key/kernel/rms
K:I 2;RMSprop/transformer_block/multi_head_attention/key/bias/rms
S:Q  2?RMSprop/transformer_block/multi_head_attention/value/kernel/rms
M:K 2=RMSprop/transformer_block/multi_head_attention/value/bias/rms
^:\  2JRMSprop/transformer_block/multi_head_attention/attention_output/kernel/rms
T:R 2HRMSprop/transformer_block/multi_head_attention/attention_output/bias/rms
::8  2*RMSprop/transformer_block/dense/kernel/rms
4:2 2(RMSprop/transformer_block/dense/bias/rms
<::  2,RMSprop/transformer_block/dense_1/kernel/rms
6:4 2*RMSprop/transformer_block/dense_1/bias/rms
C:A 27RMSprop/transformer_block/layer_normalization/gamma/rms
B:@ 26RMSprop/transformer_block/layer_normalization/beta/rms
E:C 29RMSprop/transformer_block/layer_normalization_1/gamma/rms
D:B 28RMSprop/transformer_block/layer_normalization_1/beta/rms
?2?
@__inference_model_layer_call_and_return_conditional_losses_16819
@__inference_model_layer_call_and_return_conditional_losses_17462
@__inference_model_layer_call_and_return_conditional_losses_17313
@__inference_model_layer_call_and_return_conditional_losses_16754?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_16158?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *D?A
??<
?
input_12o
?
input_22o
?2?
%__inference_model_layer_call_fn_17058
%__inference_model_layer_call_fn_17570
%__inference_model_layer_call_fn_17516
%__inference_model_layer_call_fn_16939?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_conv1d_layer_call_and_return_conditional_losses_17586?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv1d_layer_call_fn_17595?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_17611?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv1d_1_layer_call_fn_17620?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_16167?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
-__inference_max_pooling1d_layer_call_fn_16173?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_16182?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
/__inference_max_pooling1d_1_layer_call_fn_16188?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_17632
B__inference_dropout_layer_call_and_return_conditional_losses_17637?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_17642
'__inference_dropout_layer_call_fn_17647?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_17664
D__inference_dropout_1_layer_call_and_return_conditional_losses_17659?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_1_layer_call_fn_17669
)__inference_dropout_1_layer_call_fn_17674?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_17681?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_17687?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_transformer_block_layer_call_and_return_conditional_losses_17898
L__inference_transformer_block_layer_call_and_return_conditional_losses_17803?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_transformer_block_layer_call_fn_17935
1__inference_transformer_block_layer_call_fn_17972?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_17978?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_17983?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_4_layer_call_and_return_conditional_losses_17995
D__inference_dropout_4_layer_call_and_return_conditional_losses_18000?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_4_layer_call_fn_18010
)__inference_dropout_4_layer_call_fn_18005?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_18021?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_18030?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_18041?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_3_layer_call_fn_18050?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_17122input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_16158?YZ[\]^_`efabcdghHINON?K
D?A
??<
?
input_12o
?
input_22o
? "(?%
#
dense_3?
dense_32?
F__inference_concatenate_layer_call_and_return_conditional_losses_17681tP?M
F?C
A?>
?
inputs/02"
?
inputs/12"
? " ?
?
02" 
? ?
+__inference_concatenate_layer_call_fn_17687gP?M
F?C
A?>
?
inputs/02"
?
inputs/12"
? "?2" ?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_17611R*?'
 ?
?
inputs2o
? " ?
?
02f
? q
(__inference_conv1d_1_layer_call_fn_17620E*?'
 ?
?
inputs2o
? "?2f?
A__inference_conv1d_layer_call_and_return_conditional_losses_17586R*?'
 ?
?
inputs2o
? " ?
?
02f
? o
&__inference_conv1d_layer_call_fn_17595E*?'
 ?
?
inputs2o
? "?2f?
B__inference_dense_2_layer_call_and_return_conditional_losses_18021KHI'?$
?
?
inputs	2?
? "?
?
02 
? i
'__inference_dense_2_layer_call_fn_18030>HI'?$
?
?
inputs	2?
? "?2 ?
B__inference_dense_3_layer_call_and_return_conditional_losses_18041JNO&?#
?
?
inputs2 
? "?
?
02
? h
'__inference_dense_3_layer_call_fn_18050=NO&?#
?
?
inputs2 
? "?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_17659R.?+
$?!
?
inputs2"
p
? " ?
?
02"
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_17664R.?+
$?!
?
inputs2"
p 
? " ?
?
02"
? r
)__inference_dropout_1_layer_call_fn_17669E.?+
$?!
?
inputs2"
p
? "?2"r
)__inference_dropout_1_layer_call_fn_17674E.?+
$?!
?
inputs2"
p 
? "?2"?
D__inference_dropout_4_layer_call_and_return_conditional_losses_17995L+?(
!?
?
inputs	2?
p
? "?
?
0	2?
? ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_18000L+?(
!?
?
inputs	2?
p 
? "?
?
0	2?
? l
)__inference_dropout_4_layer_call_fn_18005?+?(
!?
?
inputs	2?
p
? "?	2?l
)__inference_dropout_4_layer_call_fn_18010?+?(
!?
?
inputs	2?
p 
? "?	2??
B__inference_dropout_layer_call_and_return_conditional_losses_17632R.?+
$?!
?
inputs2"
p
? " ?
?
02"
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_17637R.?+
$?!
?
inputs2"
p 
? " ?
?
02"
? p
'__inference_dropout_layer_call_fn_17642E.?+
$?!
?
inputs2"
p
? "?2"p
'__inference_dropout_layer_call_fn_17647E.?+
$?!
?
inputs2"
p 
? "?2"?
B__inference_flatten_layer_call_and_return_conditional_losses_17978K*?'
 ?
?
inputs2" 
? "?
?
0	2?
? i
'__inference_flatten_layer_call_fn_17983>*?'
 ?
?
inputs2" 
? "?	2??
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_16182?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
/__inference_max_pooling1d_1_layer_call_fn_16188wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_16167?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
-__inference_max_pooling1d_layer_call_fn_16173wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
@__inference_model_layer_call_and_return_conditional_losses_16754?YZ[\]^_`efabcdghHINOV?S
L?I
??<
?
input_12o
?
input_22o
p

 
? "?
?
02
? ?
@__inference_model_layer_call_and_return_conditional_losses_16819?YZ[\]^_`efabcdghHINOV?S
L?I
??<
?
input_12o
?
input_22o
p 

 
? "?
?
02
? ?
@__inference_model_layer_call_and_return_conditional_losses_17313?YZ[\]^_`efabcdghHINOX?U
N?K
A?>
?
inputs/02o
?
inputs/12o
p

 
? "?
?
02
? ?
@__inference_model_layer_call_and_return_conditional_losses_17462?YZ[\]^_`efabcdghHINOX?U
N?K
A?>
?
inputs/02o
?
inputs/12o
p 

 
? "?
?
02
? ?
%__inference_model_layer_call_fn_16939?YZ[\]^_`efabcdghHINOV?S
L?I
??<
?
input_12o
?
input_22o
p

 
? "?2?
%__inference_model_layer_call_fn_17058?YZ[\]^_`efabcdghHINOV?S
L?I
??<
?
input_12o
?
input_22o
p 

 
? "?2?
%__inference_model_layer_call_fn_17516?YZ[\]^_`efabcdghHINOX?U
N?K
A?>
?
inputs/02o
?
inputs/12o
p

 
? "?2?
%__inference_model_layer_call_fn_17570?YZ[\]^_`efabcdghHINOX?U
N?K
A?>
?
inputs/02o
?
inputs/12o
p 

 
? "?2?
#__inference_signature_wrapper_17122?YZ[\]^_`efabcdghHINO_?\
? 
U?R
'
input_1?
input_12o
'
input_2?
input_22o"(?%
#
dense_3?
dense_32?
L__inference_transformer_block_layer_call_and_return_conditional_losses_17803dYZ[\]^_`efabcdgh.?+
$?!
?
inputs2" 
p
? " ?
?
02" 
? ?
L__inference_transformer_block_layer_call_and_return_conditional_losses_17898dYZ[\]^_`efabcdgh.?+
$?!
?
inputs2" 
p 
? " ?
?
02" 
? ?
1__inference_transformer_block_layer_call_fn_17935WYZ[\]^_`efabcdgh.?+
$?!
?
inputs2" 
p
? "?2" ?
1__inference_transformer_block_layer_call_fn_17972WYZ[\]^_`efabcdgh.?+
$?!
?
inputs2" 
p 
? "?2" 