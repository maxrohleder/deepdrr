import numpy as np

# data taken from McGPU spectra files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[MEAN FREE PATHS (cm)]
#[Energy (eV) 	| Rayleigh 	| Compton 	| Photoelectric 	| TOTAL (+pair prod) (cm) | Rayleigh: max cumul prob F^2]

adipose_ICRP110_MFP = np.array([
	[1.5000000000E+04, 1.0177370184E+01, 5.8563234204E+00, 1.2927738195E+00, 9.5919275290E-01, 7.1184577206E-01],
	[1.5500000000E+04, 1.0646255624E+01, 5.8296293892E+00, 1.4365519561E+00, 1.0399567022E+00, 7.2591857754E-01],
	[1.6000000000E+04, 1.1125716334E+01, 5.8050484633E+00, 1.5909790057E+00, 1.1227256963E+00, 7.3946765957E-01],
	[1.6500000000E+04, 1.1615970303E+01, 5.7824175196E+00, 1.7564731796E+00, 1.2072204934E+00, 7.5248939885E-01],
	[1.7000000000E+04, 1.2117217773E+01, 5.7615869470E+00, 1.9334554459E+00, 1.2931605266E+00, 7.6498372848E-01],
	[1.7500000000E+04, 1.2629639081E+01, 5.7424189324E+00, 2.1223494620E+00, 1.3802671911E+00, 7.7695297886E-01],
	[1.8000000000E+04, 1.3153372733E+01, 5.7247856776E+00, 2.3251496294E+00, 1.4688925181E+00, 7.8840288444E-01],
	[1.8500000000E+04, 1.3688607380E+01, 5.7085913942E+00, 2.5409603205E+00, 1.5581663339E+00, 7.9934133455E-01],
	[1.9000000000E+04, 1.4235690704E+01, 5.6937801548E+00, 2.7702300496E+00, 1.6478344674E+00, 8.0977850527E-01],
	[1.9500000000E+04, 1.4794293677E+01, 5.6801846016E+00, 3.0134101795E+00, 1.7376370047E+00, 8.1972738718E-01],
	[2.0000000000E+04, 1.5364780816E+01, 5.6677689176E+00, 3.2709548619E+00, 1.8273474510E+00, 8.2920114445E-01],
	[2.0500000000E+04, 1.5947275976E+01, 5.6564622578E+00, 3.5437917449E+00, 1.9168868355E+00, 8.3821383042E-01],
	[2.1000000000E+04, 1.6541560866E+01, 5.6461503940E+00, 3.8319740602E+00, 2.0059065418E+00, 8.4678034917E-01],
	[2.1500000000E+04, 1.7148042921E+01, 5.6368270336E+00, 4.1359680330E+00, 2.0942328211E+00, 8.5491719826E-01],
	[2.2000000000E+04, 1.7766385661E+01, 5.6283740880E+00, 4.4562425525E+00, 2.1816837530E+00, 8.6264092928E-01],
	[2.2500000000E+04, 1.8396778115E+01, 5.6207613908E+00, 4.7937117494E+00, 2.2682124841E+00, 8.6996756977E-01],
	[2.3000000000E+04, 1.9039368579E+01, 5.6139546928E+00, 5.1496807228E+00, 2.3538434071E+00, 8.7691337702E-01],
	[2.3500000000E+04, 1.9693752261E+01, 5.6078448809E+00, 5.5235606510E+00, 2.4381812104E+00, 8.8349444675E-01],
	[2.4000000000E+04, 2.0360509583E+01, 5.6024679958E+00, 5.9158423225E+00, 2.5211455760E+00, 8.8972690312E-01],
	[2.4500000000E+04, 2.1039390348E+01, 5.5977439665E+00, 6.3270191932E+00, 2.6026376041E+00, 8.9562655511E-01],
	[2.5000000000E+04, 2.1730238079E+01, 5.5936116881E+00, 6.7575873437E+00, 2.6825761090E+00, 9.0120898704E-01],
	[2.5500000000E+04, 2.2433147031E+01, 5.5900477755E+00, 7.2093868153E+00, 2.7611015472E+00, 9.0648941402E-01],
	[2.6000000000E+04, 2.3148595450E+01, 5.5870813806E+00, 7.6817251660E+00, 2.8379857787E+00, 9.1148261534E-01],
	[2.6500000000E+04, 2.3876067836E+01, 5.5846136987E+00, 8.1751136069E+00, 2.9131687729E+00, 9.1620299572E-01],
	[2.7000000000E+04, 2.4615576097E+01, 5.5826189681E+00, 8.6900659500E+00, 2.9866258399E+00, 9.2066438987E-01],
	[2.7500000000E+04, 2.5367127866E+01, 5.5810728913E+00, 9.2271186998E+00, 3.0583442625E+00, 9.2488035221E-01],
	[2.8000000000E+04, 2.6130726632E+01, 5.5799525247E+00, 9.7868447880E+00, 3.1283229197E+00, 9.2886345947E-01],
	[2.8500000000E+04, 2.6906371876E+01, 5.5792361796E+00, 1.0369701481E+01, 3.1965557659E+00, 9.3262609044E-01],
	[2.9000000000E+04, 2.7694237527E+01, 5.5789250252E+00, 1.0976213150E+01, 3.2630611565E+00, 9.3618038438E-01],
	[2.9500000000E+04, 2.8494140023E+01, 5.5789778299E+00, 1.1606906606E+01, 3.3278441723E+00, 9.3953739661E-01],
	[3.0000000000E+04, 2.9306020574E+01, 5.5793705581E+00, 1.2262311065E+01, 3.3909217579E+00, 9.4270784487E-01],
	[3.0500000000E+04, 3.0129866690E+01, 5.5800857257E+00, 1.2946487575E+01, 3.4525689221E+00, 9.4570224176E-01],
	[3.1000000000E+04, 3.0965668478E+01, 5.5811061562E+00, 1.3656766573E+01, 3.5125487931E+00, 9.4853021272E-01],
	[3.1500000000E+04, 3.1813415357E+01, 5.5824153423E+00, 1.4393698268E+01, 3.5708921925E+00, 9.5120098420E-01],
	[3.2000000000E+04, 3.2673094156E+01, 5.5839976146E+00, 1.5157835365E+01, 3.6276325908E+00, 9.5372354672E-01],
	[3.2500000000E+04, 3.3544689268E+01, 5.5858380903E+00, 1.5949770522E+01, 3.6828075898E+00, 9.5610613473E-01],
	[3.3000000000E+04, 3.4428182801E+01, 5.5879226245E+00, 1.6770191818E+01, 3.7364605267E+00, 9.5835665456E-01],
	[3.3500000000E+04, 3.5323554740E+01, 5.5902377657E+00, 1.7619510041E+01, 3.7886213478E+00, 9.6048271555E-01],
	[3.4000000000E+04, 3.6230783122E+01, 5.5927707143E+00, 1.8498288428E+01, 3.8393294915E+00, 9.6249129291E-01],
	[3.4500000000E+04, 3.7149994384E+01, 5.5955254111E+00, 1.9407092593E+01, 3.8886344080E+00, 9.6438913375E-01],
	[3.5000000000E+04, 3.8081130031E+01, 5.5984863383E+00, 2.0346490498E+01, 3.9365745565E+00, 9.6618260171E-01],
	[3.5500000000E+04, 3.9024084442E+01, 5.6016335760E+00, 2.1318624492E+01, 3.9832401978E+00, 9.6787760192E-01],
	[3.6000000000E+04, 3.9978839309E+01, 5.6049566735E+00, 2.2322619730E+01, 4.0286135823E+00, 9.6947984774E-01],
	[3.6500000000E+04, 4.0945376565E+01, 5.6084456846E+00, 2.3359055671E+01, 4.0727349434E+00, 9.7099461923E-01],
	[3.7000000000E+04, 4.1923677122E+01, 5.6120911068E+00, 2.4428514088E+01, 4.1156440141E+00, 9.7242693447E-01],
	[3.7500000000E+04, 4.2913720705E+01, 5.6158838502E+00, 2.5531579046E+01, 4.1573798916E+00, 9.7378155503E-01],
	[3.8000000000E+04, 4.3915501390E+01, 5.6198167318E+00, 2.6668836875E+01, 4.1979819183E+00, 9.7506289287E-01],
	[3.8500000000E+04, 4.4929486075E+01, 5.6239290057E+00, 2.7840876151E+01, 4.2375190204E+00, 9.7627523262E-01],
	[3.9000000000E+04, 4.5955181426E+01, 5.6281660212E+00, 2.9048287676E+01, 4.2759978537E+00, 9.7742249189E-01],
	[3.9500000000E+04, 4.6992562304E+01, 5.6325200228E+00, 3.0291664455E+01, 4.3134540366E+00, 9.7850828194E-01],
	[4.0000000000E+04, 4.8041603266E+01, 5.6369835698E+00, 3.1571601676E+01, 4.3499221929E+00, 9.7953612786E-01],
	[4.0500000000E+04, 4.9102325357E+01, 5.6415538647E+00, 3.2891903075E+01, 4.3854959254E+00, 9.8050943870E-01],
	[4.1000000000E+04, 5.0175305981E+01, 5.6462784761E+00, 3.4250187178E+01, 4.4201854729E+00, 9.8143136772E-01],
	[4.1500000000E+04, 5.1259915330E+01, 5.6510935681E+00, 3.5647063046E+01, 4.4539834340E+00, 9.8230472500E-01],
	[4.2000000000E+04, 5.2356130624E+01, 5.6559926446E+00, 3.7083141932E+01, 4.4869203071E+00, 9.8313222998E-01],
	[4.2500000000E+04, 5.3463928368E+01, 5.6609694937E+00, 3.8559037265E+01, 4.5190255443E+00, 9.8391652070E-01],
	[4.3000000000E+04, 5.4584115310E+01, 5.6660890511E+00, 4.0075364633E+01, 4.5503790483E+00, 9.8466010627E-01],
	[4.3500000000E+04, 5.5715881538E+01, 5.6712766499E+00, 4.1632741761E+01, 4.5809590191E+00, 9.8536522355E-01],
	[4.4000000000E+04, 5.6859193435E+01, 5.6765261099E+00, 4.3231788496E+01, 4.6107912881E+00, 9.8603398589E-01],
	[4.4500000000E+04, 5.8014407265E+01, 5.6818633118E+00, 4.4873126789E+01, 4.6399244090E+00, 9.8666844538E-01],
	[4.5000000000E+04, 5.9181711428E+01, 5.6872992954E+00, 4.6557380678E+01, 4.6683953772E+00, 9.8727056762E-01],
	[4.5500000000E+04, 6.0360519111E+01, 5.6927829752E+00, 4.8287692988E+01, 4.6962172090E+00, 9.8784212662E-01],
	[4.6000000000E+04, 6.1550944661E+01, 5.6983207274E+00, 5.0062333198E+01, 4.7233956000E+00, 9.8838477850E-01],
	[4.6500000000E+04, 6.2753776832E+01, 5.7039701505E+00, 5.1881936186E+01, 4.7500000726E+00, 9.8890012974E-01],
	[4.7000000000E+04, 6.3968076181E+01, 5.7096551105E+00, 5.3747138904E+01, 4.7759969874E+00, 9.8938971882E-01],
	[4.7500000000E+04, 6.5193929026E+01, 5.7153799991E+00, 5.5658580360E+01, 4.8014129078E+00, 9.8985493976E-01],
	[4.8000000000E+04, 6.6432294988E+01, 5.7212128281E+00, 5.7616901603E+01, 4.8263234944E+00, 9.9029709894E-01],
	[4.8500000000E+04, 6.7682085452E+01, 5.7270701351E+00, 5.9622745708E+01, 4.8506854220E+00, 9.9071746182E-01],
	[4.9000000000E+04, 6.8943583119E+01, 5.7329704195E+00, 6.1676757762E+01, 4.8745338370E+00, 9.9111723314E-01],
	[4.9500000000E+04, 7.0217470585E+01, 5.7389592028E+00, 6.3779584848E+01, 4.8979250064E+00, 9.9149750392E-01],
	[5.0000000000E+04, 7.1502732411E+01, 5.7449623134E+00, 6.5931876032E+01, 4.9208192058E+00, 9.9185931090E-01],
	[5.0500000000E+04, 7.2800085333E+01, 5.7510273028E+00, 6.8136056718E+01, 4.9432822086E+00, 9.9220365790E-01],
	[5.1000000000E+04, 7.4109463828E+01, 5.7571468071E+00, 7.0391105552E+01, 4.9653161519E+00, 9.9253151703E-01],
	[5.1500000000E+04, 7.5430160000E+01, 5.7632713057E+00, 7.2697680121E+01, 4.9868979000E+00, 9.9284372619E-01],
	[5.2000000000E+04, 7.6763575106E+01, 5.7694912388E+00, 7.5056439965E+01, 5.0081174837E+00, 9.9314107666E-01],
	[5.2500000000E+04, 7.8108399084E+01, 5.7757182250E+00, 7.7468046562E+01, 5.0289185759E+00, 9.9342433994E-01],
	[5.3000000000E+04, 7.9465176117E+01, 5.7819858037E+00, 7.9933163319E+01, 5.0493436656E+00, 9.9369426280E-01],
	[5.3500000000E+04, 8.0834156430E+01, 5.7883075358E+00, 8.2452455554E+01, 5.0694185635E+00, 9.9395156366E-01],
	[5.4000000000E+04, 8.2214372672E+01, 5.7946211557E+00, 8.5026590486E+01, 5.0891055076E+00, 9.9419692987E-01],
	[5.4500000000E+04, 8.3607533967E+01, 5.8010302353E+00, 8.7656237223E+01, 5.1085041389E+00, 9.9443098369E-01],
	[5.5000000000E+04, 8.5011890268E+01, 5.8074257120E+00, 9.0342066747E+01, 5.1275362935E+00, 9.9465427355E-01],
	[5.5500000000E+04, 8.6428629433E+01, 5.8138777281E+00, 9.3084751906E+01, 5.1462730451E+00, 9.9486733157E-01],
	[5.6000000000E+04, 8.7857153669E+01, 5.8203488071E+00, 9.5884967397E+01, 5.1646947647E+00, 9.9507067535E-01],
	[5.6500000000E+04, 8.9297536281E+01, 5.8268417994E+00, 9.8743389759E+01, 5.1828151758E+00, 9.9526480519E-01],
	[5.7000000000E+04, 9.0750242653E+01, 5.8333819114E+00, 1.0166069736E+02, 5.2006666501E+00, 9.9545020221E-01],
	[5.7500000000E+04, 9.2214343595E+01, 5.8399145919E+00, 1.0463757038E+02, 5.2182132121E+00, 9.9562732256E-01],
	[5.8000000000E+04, 9.3691242491E+01, 5.8465174415E+00, 1.0767469081E+02, 5.2355312775E+00, 9.9579656806E-01],
	[5.8500000000E+04, 9.5179180100E+01, 5.8530909961E+00, 1.1077274244E+02, 5.2525452913E+00, 9.9595831340E-01],
	[5.9000000000E+04, 9.6680234047E+01, 5.8597481334E+00, 1.1393241083E+02, 5.2693616360E+00, 9.9611292376E-01],
	[5.9500000000E+04, 9.8192327980E+01, 5.8663744143E+00, 1.1715438332E+02, 5.2858906260E+00, 9.9626075298E-01],
	[6.0000000000E+04, 9.9717293734E+01, 5.8730670281E+00, 1.2043934902E+02, 5.3022249040E+00, 9.9640214218E-01],
	[6.0500000000E+04, 1.0125354451E+02, 5.8797404397E+00, 1.2379210662E+02, 5.3183061604E+00, 9.9653742503E-01],
	[6.1000000000E+04, 1.0280249380E+02, 5.8864676014E+00, 1.2720943431E+02, 5.3341978404E+00, 9.9666692157E-01],
	[6.1500000000E+04, 1.0436289861E+02, 5.8931829187E+00, 1.3069203075E+02, 5.3498504172E+00, 9.9679088351E-01],
	[6.2000000000E+04, 1.0593590265E+02, 5.8999438938E+00, 1.3424059631E+02, 5.3653215536E+00, 9.9690955431E-01],
	[6.2500000000E+04, 1.0752045539E+02, 5.9066962048E+00, 1.3785583313E+02, 5.3805707305E+00, 9.9702317380E-01],
	[6.3000000000E+04, 1.0911758450E+02, 5.9134903000E+00, 1.4153844511E+02, 5.3956491232E+00, 9.9713197657E-01],
	[6.3500000000E+04, 1.1072627555E+02, 5.9202748630E+00, 1.4528913783E+02, 5.4105180974E+00, 9.9723619071E-01],
	[6.4000000000E+04, 1.1234759912E+02, 5.9271014142E+00, 1.4910861862E+02, 5.4252294873E+00, 9.9733603682E-01],
	[6.4500000000E+04, 1.1398041536E+02, 5.9339136410E+00, 1.5299759652E+02, 5.4397395428E+00, 9.9743172743E-01],
	[6.5000000000E+04, 1.1562600176E+02, 5.9407720110E+00, 1.5695678225E+02, 5.4541077780E+00, 9.9752346652E-01],
	[6.5500000000E+04, 1.1728292669E+02, 5.9476074595E+00, 1.6098688823E+02, 5.4682784466E+00, 9.9761144928E-01],
	[6.6000000000E+04, 1.1895284323E+02, 5.9544970359E+00, 1.6508862856E+02, 5.4823256348E+00, 9.9769586204E-01],
	[6.6500000000E+04, 1.2063385699E+02, 5.9613514027E+00, 1.6926271901E+02, 5.4961748434E+00, 9.9777687326E-01],
	[6.7000000000E+04, 1.2232816988E+02, 5.9682715967E+00, 1.7350987701E+02, 5.5099214915E+00, 9.9785462352E-01],
	[6.7500000000E+04, 1.2403341675E+02, 5.9751482829E+00, 1.7783082166E+02, 5.5234724993E+00, 9.9792924890E-01],
	[6.8000000000E+04, 1.2575202368E+02, 5.9820909546E+00, 1.8222627369E+02, 5.5369308383E+00, 9.9800088340E-01],
	[6.8500000000E+04, 1.2748176191E+02, 5.9889982769E+00, 1.8669695546E+02, 5.5502099465E+00, 9.9806965815E-01],
	[6.9000000000E+04, 1.2922444222E+02, 5.9959505168E+00, 1.9124359098E+02, 5.5633864613E+00, 9.9813570081E-01],
	[6.9500000000E+04, 1.3097873589E+02, 6.0028879155E+00, 1.9586690586E+02, 5.5764107062E+00, 9.9819913511E-01],
	[7.0000000000E+04, 1.3274545845E+02, 6.0098458245E+00, 2.0056762734E+02, 5.5893186581E+00, 9.9826008052E-01],
	[7.0500000000E+04, 1.3452436977E+02, 6.0168127510E+00, 2.0534702874E+02, 5.6021042734E+00, 9.9831865203E-01],
	[7.1000000000E+04, 1.3631509885E+02, 6.0237725215E+00, 2.1020531895E+02, 5.6147562059E+00, 9.9837496002E-01],
	[7.1500000000E+04, 1.3811868341E+02, 6.0307683755E+00, 2.1514323045E+02, 5.6273174064E+00, 9.9842910966E-01],
	[7.2000000000E+04, 1.3993337951E+02, 6.0377262990E+00, 2.2016149731E+02, 5.6397240502E+00, 9.9848119252E-01],
	[7.2500000000E+04, 1.4176168852E+02, 6.0447504638E+00, 2.2526085519E+02, 5.6520744397E+00, 9.9853129058E-01],
	[7.3000000000E+04, 1.4360107322E+02, 6.0517344399E+00, 2.3044204130E+02, 5.6642748236E+00, 9.9857948445E-01],
	[7.3500000000E+04, 1.4545339162E+02, 6.0587548105E+00, 2.3570579441E+02, 5.6763982762E+00, 9.9862585314E-01],
	[7.4000000000E+04, 1.4731758077E+02, 6.0657670130E+00, 2.4105285486E+02, 5.6884071519E+00, 9.9867047372E-01],
	[7.4500000000E+04, 1.4919379545E+02, 6.0727773397E+00, 2.4648396452E+02, 5.7003101267E+00, 9.9871342106E-01],
	[7.5000000000E+04, 1.5108283649E+02, 6.0798171794E+00, 2.5199986681E+02, 5.7121388790E+00, 9.9875476764E-01],
	[7.5500000000E+04, 1.5298300555E+02, 6.0868183222E+00, 2.5760130666E+02, 5.7238335263E+00, 9.9879458345E-01],
	[7.6000000000E+04, 1.5489683763E+02, 6.0938810050E+00, 2.6328903056E+02, 5.7354888884E+00, 9.9883293582E-01],
	[7.6500000000E+04, 1.5682193489E+02, 6.1009096937E+00, 2.6906378649E+02, 5.7470196835E+00, 9.9886989305E-01],
	[7.7000000000E+04, 1.5875957794E+02, 6.1079546738E+00, 2.7492632395E+02, 5.7584746865E+00, 9.9890552262E-01],
	[7.7500000000E+04, 1.6070964699E+02, 6.1150103438E+00, 2.8087739393E+02, 5.7698511911E+00, 9.9893987055E-01],
	[7.8000000000E+04, 1.6267104796E+02, 6.1220344829E+00, 2.8691774894E+02, 5.7811125090E+00, 9.9897298166E-01],
	[7.8500000000E+04, 1.6464613046E+02, 6.1291165853E+00, 2.9304814297E+02, 5.7923436524E+00, 9.9900490077E-01],
	[7.9000000000E+04, 1.6663242477E+02, 6.1361618366E+00, 2.9926933149E+02, 5.8034591143E+00, 9.9903567237E-01],
	[7.9500000000E+04, 1.6863137082E+02, 6.1432248394E+00, 3.0558207145E+02, 5.8145115461E+00, 9.9906534041E-01],
	[8.0000000000E+04, 1.7064274886E+02, 6.1502964334E+00, 3.1198712125E+02, 5.8254945488E+00, 9.9909394802E-01],
	[8.0500000000E+04, 1.7266536156E+02, 6.1573320317E+00, 3.1848569577E+02, 5.8363688380E+00, 9.9912153739E-01],
	[8.1000000000E+04, 1.7470184958E+02, 6.1644291013E+00, 3.2507811813E+02, 5.8472267468E+00, 9.9914814959E-01],
	[8.1500000000E+04, 1.7674961365E+02, 6.1714910004E+00, 3.3176515140E+02, 5.8579803225E+00, 9.9917382452E-01],
	[8.2000000000E+04, 1.7880970423E+02, 6.1785565582E+00, 3.3854756012E+02, 5.8686674543E+00, 9.9919860077E-01],
	[8.2500000000E+04, 1.8088265379E+02, 6.1856443744E+00, 3.4542611028E+02, 5.8793072553E+00, 9.9922251564E-01],
	[8.3000000000E+04, 1.8296689178E+02, 6.1926974210E+00, 3.5240156929E+02, 5.8898482395E+00, 9.9924560504E-01],
	[8.3500000000E+04, 1.8506445457E+02, 6.1997889969E+00, 3.5947470597E+02, 5.9003604380E+00, 9.9926790355E-01],
	[8.4000000000E+04, 1.8717394041E+02, 6.2068679906E+00, 3.6664629060E+02, 5.9107977571E+00, 9.9928944435E-01],
	[8.4500000000E+04, 1.8929498652E+02, 6.2139217823E+00, 3.7391709485E+02, 5.9211498982E+00, 9.9931025927E-01],
	[8.5000000000E+04, 1.9142975003E+02, 6.2210263547E+00, 3.8128789182E+02, 5.9314894556E+00, 9.9933037881E-01],
	[8.5500000000E+04, 1.9357585563E+02, 6.2280973093E+00, 3.8875945598E+02, 5.9417389101E+00, 9.9934983147E-01],
	[8.6000000000E+04, 1.9573428736E+02, 6.2351695040E+00, 3.9633256325E+02, 5.9519322817E+00, 9.9936863980E-01],
	[8.6500000000E+04, 1.9790572837E+02, 6.2422660532E+00, 4.0400799089E+02, 5.9620926425E+00, 9.9938682462E-01],
	[8.7000000000E+04, 2.0008853500E+02, 6.2493297071E+00, 4.1178651759E+02, 5.9721676063E+00, 9.9940440680E-01],
	[8.7500000000E+04, 2.0228431951E+02, 6.2564162790E+00, 4.1966892341E+02, 5.9822108431E+00, 9.9942140707E-01],
	[8.8000000000E+04, 2.0449251689E+02, 6.2635055296E+00, 4.2765598976E+02, 5.9922046276E+00, 9.9943784588E-01],
	[8.8500000000E+04, 2.0671210355E+02, 6.2705625790E+00, 4.3574849945E+02, 6.0021173541E+00, 9.9945374340E-01],
	[8.9000000000E+04, 2.0894520233E+02, 6.2776596022E+00, 4.4394723666E+02, 6.0120181039E+00, 9.9946911936E-01],
	[8.9500000000E+04, 2.1119023262E+02, 6.2847422377E+00, 4.5225298691E+02, 6.0218568857E+00, 9.9948399307E-01],
	[9.0000000000E+04, 2.1344667597E+02, 6.2917933395E+00, 4.6066653708E+02, 6.0316186170E+00, 9.9949838329E-01],
	[9.0500000000E+04, 2.1571704446E+02, 6.2988969666E+00, 4.6917631018E+02, 6.0413815778E+00, 9.9951230831E-01],
	[9.1000000000E+04, 2.1799898114E+02, 6.3059736142E+00, 4.7779507972E+02, 6.0510739913E+00, 9.9952578581E-01],
	[9.1500000000E+04, 2.2029249505E+02, 6.3130240185E+00, 4.8652363111E+02, 6.0606975208E+00, 9.9953883292E-01],
	[9.2000000000E+04, 2.2259994032E+02, 6.3201257718E+00, 4.9536275105E+02, 6.0703264500E+00, 9.9955146621E-01],
	[9.2500000000E+04, 2.2491885171E+02, 6.3271969572E+00, 5.0431322756E+02, 6.0798843695E+00, 9.9956370161E-01],
	[9.3000000000E+04, 2.2724950689E+02, 6.3342471253E+00, 5.1337584995E+02, 6.0893811983E+00, 9.9957555452E-01],
	[9.3500000000E+04, 2.2959396824E+02, 6.3413432876E+00, 5.2255140879E+02, 6.0988812584E+00, 9.9958704048E-01],
	[9.4000000000E+04, 2.3194991922E+02, 6.3484094897E+00, 5.3184069599E+02, 6.1083135688E+00, 9.9959817911E-01],
	[9.4500000000E+04, 2.3431765601E+02, 6.3554557498E+00, 5.4124450471E+02, 6.1176884355E+00, 9.9960898084E-01],
	[9.5000000000E+04, 2.3669919554E+02, 6.3625467929E+00, 5.5076362937E+02, 6.1270680448E+00, 9.9961945420E-01],
	[9.5500000000E+04, 2.3909224811E+02, 6.3696084620E+00, 5.6039886571E+02, 6.1363829322E+00, 9.9962960796E-01],
	[9.6000000000E+04, 2.4149700318E+02, 6.3766472403E+00, 5.7015101068E+02, 6.1456399986E+00, 9.9963945109E-01],
	[9.6500000000E+04, 2.4391568011E+02, 6.3837336081E+00, 5.8002086255E+02, 6.1549069197E+00, 9.9964899265E-01],
	[9.7000000000E+04, 2.4634589337E+02, 6.3907911677E+00, 5.9000922080E+02, 6.1641119350E+00, 9.9965824175E-01],
	[9.7500000000E+04, 2.4878763812E+02, 6.3978201840E+00, 6.0011688619E+02, 6.1732559853E+00, 9.9966720748E-01],
	[9.8000000000E+04, 2.5124347074E+02, 6.4049010949E+00, 6.1034466072E+02, 6.1824162449E+00, 9.9967589888E-01],
	[9.8500000000E+04, 2.5371090083E+02, 6.4119549431E+00, 6.2069334763E+02, 6.1915183613E+00, 9.9968432489E-01],
	[9.9000000000E+04, 2.5618988574E+02, 6.4189807852E+00, 6.3116375141E+02, 6.2005620740E+00, 9.9969249432E-01],
	[9.9500000000E+04, 2.5868260758E+02, 6.4260466556E+00, 6.4175667778E+02, 6.2096127966E+00, 9.9970041583E-01],
	[1.0000000000E+05, 2.6118730768E+02, 6.4330971664E+00, 6.5247293369E+02, 6.2186184607E+00, 9.9970809789E-01],
	[1.0050000000E+05, 2.6370358575E+02, 6.4401201902E+00, 6.6328245899E+02, 6.2275653898E+00, 9.9971554878E-01],
	[1.0100000000E+05, 2.6623312340E+02, 6.4471677265E+00, 6.7421607071E+02, 6.2365065512E+00, 9.9972277657E-01],
	[1.0150000000E+05, 2.6877514504E+02, 6.4542152421E+00, 6.8527456818E+02, 6.2454191710E+00, 9.9972978910E-01],
	[1.0200000000E+05, 2.7132876763E+02, 6.4612357731E+00, 6.9645875187E+02, 6.2542780911E+00, 9.9973659399E-01],
	[1.0250000000E+05, 2.7389504646E+02, 6.4682617628E+00, 7.0776942345E+02, 6.2631148158E+00, 9.9974319862E-01],
	[1.0300000000E+05, 2.7647443995E+02, 6.4753065937E+00, 7.1920738569E+02, 6.2719426890E+00, 9.9974961014E-01],
	[1.0350000000E+05, 2.7906545716E+02, 6.4823249261E+00, 7.3077344255E+02, 6.2807189446E+00, 9.9975583545E-01],
	[1.0400000000E+05, 2.8166840113E+02, 6.4893262393E+00, 7.4246839912E+02, 6.2894531327E+00, 9.9976188123E-01],
	[1.0450000000E+05, 2.8428521400E+02, 6.4963686756E+00, 7.5429306163E+02, 6.2982013478E+00, 9.9976775390E-01],
	[1.0500000000E+05, 2.8691367318E+02, 6.5033850843E+00, 7.6624823744E+02, 6.3068998957E+00, 9.9977345969E-01],
	[1.0550000000E+05, 2.8955377466E+02, 6.5103756849E+00, 7.7833473505E+02, 6.3155494399E+00, 9.9977900456E-01],
	[1.0600000000E+05, 2.9220748202E+02, 6.5173989863E+00, 7.9055336410E+02, 6.3242064400E+00, 9.9978439426E-01],
	[1.0650000000E+05, 2.9487342762E+02, 6.5244137283E+00, 8.0290493534E+02, 6.3328318826E+00, 9.9978963435E-01],
	[1.0700000000E+05, 2.9755103802E+02, 6.5314031114E+00, 8.1539026064E+02, 6.3414101121E+00, 9.9979473013E-01],
	[1.0750000000E+05, 3.0024125228E+02, 6.5383950683E+00, 8.2801015300E+02, 6.3499683109E+00, 9.9979968674E-01],
	[1.0800000000E+05, 3.0294472587E+02, 6.5454083839E+00, 8.4076542654E+02, 6.3585249256E+00, 9.9980450896E-01],
	[1.0850000000E+05, 3.0565988649E+02, 6.5523967759E+00, 8.5365689647E+02, 6.3670360096E+00, 9.9980920016E-01],
	[1.0900000000E+05, 3.0838673047E+02, 6.5593604473E+00, 8.6668537911E+02, 6.3755021396E+00, 9.9981376347E-01],
	[1.0950000000E+05, 3.1112756730E+02, 6.5663666208E+00, 8.7985169190E+02, 6.3839882067E+00, 9.9981820213E-01],
	[1.1000000000E+05, 3.1388031654E+02, 6.5733542331E+00, 8.9315665336E+02, 6.3924360242E+00, 9.9982251938E-01],
	[1.1050000000E+05, 3.1664477124E+02, 6.5803175408E+00, 9.0660108313E+02, 6.4008404379E+00, 9.9982671851E-01],
	[1.1100000000E+05, 3.1942194573E+02, 6.5872860523E+00, 9.2018580192E+02, 6.4092301426E+00, 9.9983080278E-01],
	[1.1150000000E+05, 3.2221231907E+02, 6.5942730817E+00, 9.3391163154E+02, 6.4176183059E+00, 9.9983477544E-01],
	[1.1200000000E+05, 3.2501441967E+02, 6.6012362103E+00, 9.4777918317E+02, 6.4259645160E+00, 9.9983863971E-01],
	[1.1250000000E+05, 3.2782824412E+02, 6.6081756263E+00, 9.6178915732E+02, 6.4342692727E+00, 9.9984239878E-01],
	[1.1300000000E+05, 3.3065587996E+02, 6.6151509640E+00, 9.7594270041E+02, 6.4425902680E+00, 9.9984605579E-01],
	[1.1350000000E+05, 3.3349565871E+02, 6.6221140849E+00, 9.9024063837E+02, 6.4508815549E+00, 9.9984961382E-01],
	[1.1400000000E+05, 3.3634718292E+02, 6.6290538797E+00, 1.0046837983E+03, 6.4591327537E+00, 9.9985307590E-01],
	[1.1450000000E+05, 3.3921098060E+02, 6.6359855654E+00, 1.0192730082E+03, 6.4673588087E+00, 9.9985644499E-01],
	[1.1500000000E+05, 3.4208846683E+02, 6.6429488372E+00, 1.0340090973E+03, 6.4755982508E+00, 9.9985972399E-01],
	[1.1550000000E+05, 3.4497771981E+02, 6.6498891587E+00, 1.0488928958E+03, 6.4837988802E+00, 9.9986291572E-01],
	[1.1600000000E+05, 3.4787873641E+02, 6.6568067043E+00, 1.0639252350E+03, 6.4919611408E+00, 9.9986602294E-01],
	[1.1650000000E+05, 3.5079281859E+02, 6.6637381826E+00, 1.0791069472E+03, 6.5001206807E+00, 9.9986904832E-01],
	[1.1700000000E+05, 3.5371982641E+02, 6.6706791669E+00, 1.0944388659E+03, 6.5082735314E+00, 9.9987199446E-01],
	[1.1750000000E+05, 3.5665861889E+02, 6.6775977356E+00, 1.1099218254E+03, 6.5163891963E+00, 9.9987486390E-01],
	[1.1800000000E+05, 3.5960919302E+02, 6.6844940560E+00, 1.1255566613E+03, 6.5244680889E+00, 9.9987765907E-01],
	[1.1850000000E+05, 3.6257347100E+02, 6.6914216511E+00, 1.1413442100E+03, 6.5325620950E+00, 9.9988038236E-01],
	[1.1900000000E+05, 3.6555006218E+02, 6.6983413583E+00, 1.1572853092E+03, 6.5406337092E+00, 9.9988303606E-01],
	[1.1950000000E+05, 3.6853845578E+02, 6.7052391628E+00, 1.1733807975E+03, 6.5486696505E+00, 9.9988562240E-01],
	[1.2000000000E+05, 3.7153864894E+02, 6.7121152251E+00, 1.1896315144E+03, 6.5566703050E+00, 9.9988814355E-01],
])
