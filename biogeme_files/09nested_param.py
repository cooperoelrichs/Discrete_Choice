# This file has automatically been generated
# biogeme 2.3 [Wed 29 Apr 2015 11:25:34 AEST]
# <a href='http://people.epfl.ch/michel.bierlaire'>Michel Bierlaire</a>, <a href='http://transp-or.epfl.ch'>Transport and Mobility Laboratory</a>, <a href='http://www.epfl.ch'>Ecole Polytechnique F&eacute;d&eacute;rale de Lausanne (EPFL)</a>
# Thu Apr 30 10:32:29 2015</p>
#
ASC_TRAIN = Beta('ASC_TRAIN',-0.511948,-10,10,0 )

B_TIME = Beta('B_TIME',-0.898664,-10,10,0 )

B_COST = Beta('B_COST',-0.856665,-10,10,0 )

MU = Beta('MU',2.05407,1,10,0 )

ASC_CAR = Beta('ASC_CAR',-0.167156,-10,10,0 )

ASC_SM = Beta('ASC_SM',0,-10,10,1 )


## Code for the sensitivity analysis
names = ['ASC_TRAIN','B_TIME','B_COST','MU','ASC_CAR']
values = [[0.00625896,-0.00655572,-0.000262981,0.00297523,0.00367682],[-0.00655572,0.0114731,0.00268146,0.00521772,-0.00483378],[-0.000262981,0.00268146,0.00360422,0.00360891,-0.000412283],[0.00297523,0.00521772,0.00360891,0.0269627,-0.00119567],[0.00367682,-0.00483378,-0.000412283,-0.00119567,0.00297342]]
vc = Matrix(5,names,values)
BIOGEME_OBJECT.VARCOVAR = vc
