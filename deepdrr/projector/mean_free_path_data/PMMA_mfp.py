import numpy as np

# data taken from McGPU spectra files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[MEAN FREE PATHS (cm)]
#[Energy (eV) 	| Rayleigh 	| Compton 	| Photoelectric 	| TOTAL (+pair prod) (cm) | Rayleigh: max cumul prob F^2]

PMMA_MFP = np.array([
	[1.5000000000E+04, 7.8577447737E+00, 4.9295503276E+00, 1.0163448648E+00, 7.6101228674E-01, 7.1050829940E-01],
	[1.5500000000E+04, 8.2191918021E+00, 4.9011673463E+00, 1.1297016925E+00, 8.2583958163E-01, 7.2461832262E-01],
	[1.6000000000E+04, 8.5887165415E+00, 4.8750786597E+00, 1.2514913710E+00, 8.9237638529E-01, 7.3821502159E-01],
	[1.6500000000E+04, 8.9664884476E+00, 4.8510875062E+00, 1.3820480250E+00, 9.6040298600E-01, 7.5129338290E-01],
	[1.7000000000E+04, 9.3526633195E+00, 4.8290158639E+00, 1.5217081165E+00, 1.0296972968E+00, 7.6385242970E-01],
	[1.7500000000E+04, 9.7473816253E+00, 4.8087020298E+00, 1.6708103895E+00, 1.1000374767E+00, 7.7589397279E-01],
	[1.8000000000E+04, 1.0150754187E+01, 4.7899984494E+00, 1.8309219319E+00, 1.1717064102E+00, 7.8742232059E-01],
	[1.8500000000E+04, 1.0562937973E+01, 4.7727936344E+00, 2.0013507212E+00, 1.2440068809E+00, 7.9844469648E-01],
	[1.9000000000E+04, 1.0984216989E+01, 4.7570177881E+00, 2.1824555458E+00, 1.3167338228E+00, 8.0896976222E-01],
	[1.9500000000E+04, 1.1414351988E+01, 4.7424918868E+00, 2.3745975974E+00, 1.3896752151E+00, 8.1900855866E-01],
	[2.0000000000E+04, 1.1853639195E+01, 4.7291703009E+00, 2.5781404230E+00, 1.4626462232E+00, 8.2857302116E-01],
	[2.0500000000E+04, 1.2302187381E+01, 4.7169736468E+00, 2.7937553938E+00, 1.5355608942E+00, 8.3767675946E-01],
	[2.1000000000E+04, 1.2759829654E+01, 4.7057832659E+00, 3.0215465202E+00, 1.6081519819E+00, 8.4633389207E-01],
	[2.1500000000E+04, 1.3226885144E+01, 4.6955865238E+00, 3.2618866953E+00, 1.6802752964E+00, 8.5455961481E-01],
	[2.2000000000E+04, 1.3703092348E+01, 4.6862634453E+00, 3.5151510465E+00, 1.7517778126E+00, 8.6236929977E-01],
	[2.2500000000E+04, 1.4188599970E+01, 4.6777796403E+00, 3.7820855589E+00, 1.8226219323E+00, 8.6977892707E-01],
	[2.3000000000E+04, 1.4683524955E+01, 4.6700972684E+00, 4.0637623176E+00, 1.8928308536E+00, 8.7680441410E-01],
	[2.3500000000E+04, 1.5187552328E+01, 4.6631078212E+00, 4.3596758702E+00, 1.9620635014E+00, 8.8346187669E-01],
	[2.4000000000E+04, 1.5701136653E+01, 4.6568411797E+00, 4.6702197705E+00, 2.0302499993E+00, 8.8976733917E-01],
	[2.4500000000E+04, 1.6224083681E+01, 4.6512175931E+00, 4.9957898194E+00, 2.0973032127E+00, 8.9573659716E-01],
	[2.5000000000E+04, 1.6756272290E+01, 4.6461760208E+00, 5.3367840307E+00, 2.1631505677E+00, 9.0138477968E-01],
	[2.5500000000E+04, 1.7297777057E+01, 4.6416913938E+00, 5.6945524991E+00, 2.2278858590E+00, 9.0672746246E-01],
	[2.6000000000E+04, 1.7848971861E+01, 4.6377879388E+00, 6.0686520081E+00, 2.2913344133E+00, 9.1177962192E-01],
	[2.6500000000E+04, 1.8409453613E+01, 4.6343710898E+00, 6.4594922857E+00, 2.3534403554E+00, 9.1655520484E-01],
	[2.7000000000E+04, 1.8979231478E+01, 4.6314146829E+00, 6.8674852371E+00, 2.4141781950E+00, 9.2106834481E-01],
	[2.7500000000E+04, 1.9558311268E+01, 4.6288940982E+00, 7.2930449143E+00, 2.4735306202E+00, 9.2533267231E-01],
	[2.8000000000E+04, 2.0146695538E+01, 4.6267861425E+00, 7.7365874879E+00, 2.5314876994E+00, 9.2936079575E-01],
	[2.8500000000E+04, 2.0744383697E+01, 4.6250689435E+00, 8.1985312181E+00, 2.5880461102E+00, 9.3316531302E-01],
	[2.9000000000E+04, 2.1351511243E+01, 4.6237416644E+00, 8.6792964283E+00, 2.6432170083E+00, 9.3675823200E-01],
	[2.9500000000E+04, 2.1967934667E+01, 4.6227647949E+00, 9.1793054782E+00, 2.6969999467E+00, 9.4015072596E-01],
	[3.0000000000E+04, 2.2593607874E+01, 4.6221148006E+00, 9.6989827386E+00, 2.7494048046E+00, 9.4335386194E-01],
	[3.0500000000E+04, 2.3228520815E+01, 4.6217743619E+00, 1.0241685771E+01, 2.8006668620E+00, 9.4637798352E-01],
	[3.1000000000E+04, 2.3872665243E+01, 4.6217267977E+00, 1.0805183836E+01, 2.8505748965E+00, 9.4923287982E-01],
	[3.1500000000E+04, 2.4526032285E+01, 4.6219562056E+00, 1.1389918942E+01, 2.8991511724E+00, 9.5192811004E-01],
	[3.2000000000E+04, 2.5188611040E+01, 4.6224475125E+00, 1.1996335196E+01, 2.9464203825E+00, 9.5447247716E-01],
	[3.2500000000E+04, 2.5860388695E+01, 4.6231864220E+00, 1.2624878774E+01, 2.9924092127E+00, 9.5687450313E-01],
	[3.3000000000E+04, 2.6541350651E+01, 4.6241593665E+00, 1.3275997900E+01, 3.0371459514E+00, 9.5914225464E-01],
	[3.3500000000E+04, 2.7231480656E+01, 4.6253534621E+00, 1.3950142826E+01, 3.0806601390E+00, 9.6128326537E-01],
	[3.4000000000E+04, 2.7930760939E+01, 4.6267564674E+00, 1.4647765805E+01, 3.1229822567E+00, 9.6330483138E-01],
	[3.4500000000E+04, 2.8639288921E+01, 4.6283712444E+00, 1.5369321077E+01, 3.1641516504E+00, 9.6521368924E-01],
	[3.5000000000E+04, 2.9357018276E+01, 4.6301831625E+00, 1.6115264841E+01, 3.2041982737E+00, 9.6701632646E-01],
	[3.5500000000E+04, 3.0083866271E+01, 4.6321736504E+00, 1.6887125871E+01, 3.2431890281E+00, 9.6871881190E-01],
	[3.6000000000E+04, 3.0819818138E+01, 4.6343325935E+00, 1.7684378226E+01, 3.2811139801E+00, 9.7032689150E-01],
	[3.6500000000E+04, 3.1564859339E+01, 4.6366503665E+00, 1.8507486979E+01, 3.3180049935E+00, 9.7184609766E-01],
	[3.7000000000E+04, 3.2318974564E+01, 4.6391177970E+00, 1.9356919127E+01, 3.3538936525E+00, 9.7328136058E-01],
	[3.7500000000E+04, 3.3082147598E+01, 4.6417261396E+00, 2.0233143573E+01, 3.3888111435E+00, 9.7463745771E-01],
	[3.8000000000E+04, 3.3854373354E+01, 4.6444683981E+00, 2.1136631107E+01, 3.4227890104E+00, 9.7591905633E-01],
	[3.8500000000E+04, 3.4636012359E+01, 4.6473787062E+00, 2.2067854387E+01, 3.4558844429E+00, 9.7713057578E-01],
	[3.9000000000E+04, 3.5426682180E+01, 4.6504079493E+00, 2.3027287926E+01, 3.4881009794E+00, 9.7827590857E-01],
	[3.9500000000E+04, 3.6226362868E+01, 4.6535487605E+00, 2.4015408070E+01, 3.5194673408E+00, 9.7935878371E-01],
	[4.0000000000E+04, 3.7035034312E+01, 4.6567941083E+00, 2.5032692983E+01, 3.5500115201E+00, 9.8038284526E-01],
	[4.0500000000E+04, 3.7852712351E+01, 4.6601410928E+00, 2.6082173189E+01, 3.5798113754E+00, 9.8135157198E-01],
	[4.1000000000E+04, 3.8679843099E+01, 4.6636311819E+00, 2.7161959299E+01, 3.6088752427E+00, 9.8226805481E-01],
	[4.1500000000E+04, 3.9515940523E+01, 4.6672079720E+00, 2.8272540766E+01, 3.6371953938E+00, 9.8313522325E-01],
	[4.2000000000E+04, 4.0360987296E+01, 4.6708655288E+00, 2.9414408869E+01, 3.6647967664E+00, 9.8395593496E-01],
	[4.2500000000E+04, 4.1214965515E+01, 4.6745981830E+00, 3.0588056697E+01, 3.6917034814E+00, 9.8473290722E-01],
	[4.3000000000E+04, 4.2078499465E+01, 4.6784627052E+00, 3.1793979137E+01, 3.7179831398E+00, 9.8546856551E-01],
	[4.3500000000E+04, 4.2950963363E+01, 4.6823933202E+00, 3.3032672856E+01, 3.7436159225E+00, 9.8616522451E-01],
	[4.4000000000E+04, 4.3832331440E+01, 4.6863844250E+00, 3.4304636289E+01, 3.7686230533E+00, 9.8682513482E-01],
	[4.4500000000E+04, 4.4722879697E+01, 4.6904585271E+00, 3.5610369622E+01, 3.7930453742E+00, 9.8745040223E-01],
	[4.5000000000E+04, 4.5622754532E+01, 4.6946250837E+00, 3.6950374782E+01, 3.8169138354E+00, 9.8804292744E-01],
	[4.5500000000E+04, 4.6531501313E+01, 4.6988391254E+00, 3.8327007029E+01, 3.8402368921E+00, 9.8860454775E-01],
	[4.6000000000E+04, 4.7449208371E+01, 4.7031060744E+00, 3.9739036096E+01, 3.8630206701E+00, 9.8913703606E-01],
	[4.6500000000E+04, 4.8376482668E+01, 4.7074760081E+00, 4.1186972465E+01, 3.8853241324E+00, 9.8964200897E-01],
	[4.7000000000E+04, 4.9312597223E+01, 4.7118821490E+00, 4.2671328343E+01, 3.9071176515E+00, 9.9012097254E-01],
	[4.7500000000E+04, 5.0257618309E+01, 4.7163281988E+00, 4.4192617644E+01, 3.9284234309E+00, 9.9057539080E-01],
	[4.8000000000E+04, 5.1212281705E+01, 4.7208731820E+00, 4.5751355982E+01, 3.9493057381E+00, 9.9100669048E-01],
	[4.8500000000E+04, 5.2175748742E+01, 4.7254441700E+00, 4.7348060653E+01, 3.9697268633E+00, 9.9141609447E-01],
	[4.9000000000E+04, 5.3148236050E+01, 4.7300571441E+00, 4.8983250629E+01, 3.9897164597E+00, 9.9180474996E-01],
	[4.9500000000E+04, 5.4130263850E+01, 4.7347514137E+00, 5.0657446541E+01, 4.0093221824E+00, 9.9217378145E-01],
	[5.0000000000E+04, 5.5121053650E+01, 4.7394623995E+00, 5.2371170669E+01, 4.0285096367E+00, 9.9252428068E-01],
	[5.0500000000E+04, 5.6121152485E+01, 4.7442312206E+00, 5.4126042450E+01, 4.0473328314E+00, 9.9285729941E-01],
	[5.1000000000E+04, 5.7130510130E+01, 4.7490512916E+00, 5.5921553188E+01, 4.0657951832E+00, 9.9317383737E-01],
	[5.1500000000E+04, 5.8148585638E+01, 4.7538796315E+00, 5.7758231652E+01, 4.0838766518E+00, 9.9347476370E-01],
	[5.2000000000E+04, 5.9176450878E+01, 4.7587943609E+00, 5.9636608229E+01, 4.1016538092E+00, 9.9376086980E-01],
	[5.2500000000E+04, 6.0213104530E+01, 4.7637188823E+00, 6.1557214915E+01, 4.1190783084E+00, 9.9403293200E-01],
	[5.3000000000E+04, 6.1258962960E+01, 4.7686821079E+00, 6.3520585304E+01, 4.1361861863E+00, 9.9429170593E-01],
	[5.3500000000E+04, 6.2314218660E+01, 4.7736955859E+00, 6.5527254576E+01, 4.1529992098E+00, 9.9453792145E-01],
	[5.4000000000E+04, 6.3378130572E+01, 4.7787054995E+00, 6.7577759489E+01, 4.1694850036E+00, 9.9477227722E-01],
	[5.4500000000E+04, 6.4452009323E+01, 4.7838010473E+00, 6.9672638369E+01, 4.1857282915E+00, 9.9499539599E-01],
	[5.5000000000E+04, 6.5534513617E+01, 4.7888880924E+00, 7.1812431096E+01, 4.2016622965E+00, 9.9520783719E-01],
	[5.5500000000E+04, 6.6626554816E+01, 4.7940269851E+00, 7.3997679100E+01, 4.2173474139E+00, 9.9541014741E-01],
	[5.6000000000E+04, 6.7727674980E+01, 4.7991853239E+00, 7.6228925344E+01, 4.2327667624E+00, 9.9560285840E-01],
	[5.6500000000E+04, 6.8837930344E+01, 4.8043655046E+00, 7.8506714324E+01, 4.2479319168E+00, 9.9578648390E-01],
	[5.7000000000E+04, 6.9957679422E+01, 4.8095890632E+00, 8.0831592049E+01, 4.2628702700E+00, 9.9596151422E-01],
	[5.7500000000E+04, 7.1086207944E+01, 4.8148090736E+00, 8.3204106042E+01, 4.2775511403E+00, 9.9612838365E-01],
	[5.8000000000E+04, 7.2224596082E+01, 4.8200920470E+00, 8.5624805321E+01, 4.2920393693E+00, 9.9628749281E-01],
	[5.8500000000E+04, 7.3371490360E+01, 4.8253525677E+00, 8.8094240399E+01, 4.3062706726E+00, 9.9643923300E-01],
	[5.9000000000E+04, 7.4528490870E+01, 4.8306873368E+00, 9.0612963267E+01, 4.3203353271E+00, 9.9658398402E-01],
	[5.9500000000E+04, 7.5693998911E+01, 4.8359981904E+00, 9.3181527394E+01, 4.3341570983E+00, 9.9672211230E-01],
	[6.0000000000E+04, 7.6869426398E+01, 4.8413683024E+00, 9.5800487708E+01, 4.3478145618E+00, 9.9685395942E-01],
	[6.0500000000E+04, 7.8053550478E+01, 4.8467243688E+00, 9.8473204162E+01, 4.3612572386E+00, 9.9697982810E-01],
	[6.1000000000E+04, 7.9247461218E+01, 4.8521287459E+00, 1.0119756244E+02, 4.3745398777E+00, 9.9710000985E-01],
	[6.1500000000E+04, 8.0450200035E+01, 4.8575252651E+00, 1.0397412473E+02, 4.3876204250E+00, 9.9721478847E-01],
	[6.2000000000E+04, 8.1662648952E+01, 4.8629630240E+00, 1.0680345467E+02, 4.4005477283E+00, 9.9732443849E-01],
	[6.2500000000E+04, 8.2883997789E+01, 4.8683955405E+00, 1.0968611732E+02, 4.4132874015E+00, 9.9742923199E-01],
	[6.3000000000E+04, 8.4115039068E+01, 4.8738658462E+00, 1.1262267921E+02, 4.4258827753E+00, 9.9752941249E-01],
	[6.3500000000E+04, 8.5354990444E+01, 4.8793300722E+00, 1.1561370830E+02, 4.4383010901E+00, 9.9762518350E-01],
	[6.4000000000E+04, 8.6604677535E+01, 4.8848321258E+00, 1.1865977394E+02, 4.4505862136E+00, 9.9771674622E-01],
	[6.4500000000E+04, 8.7863221289E+01, 4.8903239250E+00, 1.2176144694E+02, 4.4627010866E+00, 9.9780429847E-01],
	[6.5000000000E+04, 8.9131606865E+01, 4.8958569608E+00, 1.2491929949E+02, 4.4746960500E+00, 9.9788803330E-01],
	[6.5500000000E+04, 9.0408730211E+01, 4.9013723410E+00, 1.2813390518E+02, 4.4865239326E+00, 9.9796813801E-01],
	[6.6000000000E+04, 9.1695866127E+01, 4.9069356246E+00, 1.3140583899E+02, 4.4982473635E+00, 9.9804479331E-01],
	[6.6500000000E+04, 9.2991553708E+01, 4.9124707304E+00, 1.3473567732E+02, 4.5098033645E+00, 9.9811817286E-01],
	[6.7000000000E+04, 9.4297490964E+01, 4.9180635563E+00, 1.3812399790E+02, 4.5212725448E+00, 9.9818844294E-01],
	[6.7500000000E+04, 9.5611853976E+01, 4.9236210961E+00, 1.4157137985E+02, 4.5325762850E+00, 9.9825575805E-01],
	[6.8000000000E+04, 9.6936513626E+01, 4.9292363523E+00, 1.4507840368E+02, 4.5438015145E+00, 9.9832024681E-01],
	[6.8500000000E+04, 9.8269751171E+01, 4.9348232058E+00, 1.4864565121E+02, 4.5548751770E+00, 9.9838203058E-01],
	[6.9000000000E+04, 9.9612962995E+01, 4.9404497583E+00, 1.5227370565E+02, 4.5658619239E+00, 9.9844122916E-01],
	[6.9500000000E+04, 1.0096512443E+02, 4.9460652642E+00, 1.5596315153E+02, 4.5767198687E+00, 9.9849796000E-01],
	[7.0000000000E+04, 1.0232686439E+02, 4.9516996595E+00, 1.5971457474E+02, 4.5874793356E+00, 9.9855233749E-01],
	[7.0500000000E+04, 1.0369799766E+02, 4.9573431767E+00, 1.6352852795E+02, 4.5981348807E+00, 9.9860447250E-01],
	[7.1000000000E+04, 1.0507823819E+02, 4.9629820624E+00, 1.6740563284E+02, 4.6086773254E+00, 9.9865447203E-01],
	[7.1500000000E+04, 1.0646838624E+02, 4.9686529384E+00, 1.7134647920E+02, 4.6191428773E+00, 9.9870243895E-01],
	[7.2000000000E+04, 1.0786709691E+02, 4.9742930640E+00, 1.7535165817E+02, 4.6294777751E+00, 9.9874847104E-01],
	[7.2500000000E+04, 1.0927629940E+02, 4.9799906518E+00, 1.7942176219E+02, 4.6397648336E+00, 9.9879265267E-01],
	[7.3000000000E+04, 1.1069403685E+02, 4.9856555395E+00, 1.8355738498E+02, 4.6499251070E+00, 9.9883506058E-01],
	[7.3500000000E+04, 1.1212174240E+02, 4.9913525407E+00, 1.8775912158E+02, 4.6600201385E+00, 9.9887577036E-01],
	[7.4000000000E+04, 1.1355859615E+02, 4.9970439138E+00, 1.9202756829E+02, 4.6700182514E+00, 9.9891485610E-01],
	[7.4500000000E+04, 1.1500471756E+02, 5.0027349530E+00, 1.9636332271E+02, 4.6799267549E+00, 9.9895239001E-01],
	[7.5000000000E+04, 1.1646072440E+02, 5.0084521989E+00, 2.0076698370E+02, 4.6897723331E+00, 9.9898844215E-01],
	[7.5500000000E+04, 1.1792530652E+02, 5.0141379249E+00, 2.0523915139E+02, 4.6995045989E+00, 9.9902308021E-01],
	[7.6000000000E+04, 1.1940041968E+02, 5.0198768796E+00, 2.0978042718E+02, 4.7092033515E+00, 9.9905636941E-01],
	[7.6500000000E+04, 1.2088421394E+02, 5.0255882608E+00, 2.1439141372E+02, 4.7187968538E+00, 9.9908837098E-01],
	[7.7000000000E+04, 1.2237767739E+02, 5.0313145532E+00, 2.1907271491E+02, 4.7283261446E+00, 9.9911913769E-01],
	[7.7500000000E+04, 1.2388071761E+02, 5.0370510078E+00, 2.2382493590E+02, 4.7377889678E+00, 9.9914871970E-01],
	[7.8000000000E+04, 1.2539249043E+02, 5.0427619244E+00, 2.2864868309E+02, 4.7471544787E+00, 9.9917716629E-01],
	[7.8500000000E+04, 1.2691480897E+02, 5.0485228875E+00, 2.3354456408E+02, 4.7564941723E+00, 9.9920452566E-01],
	[7.9000000000E+04, 1.2844576742E+02, 5.0542537917E+00, 2.3851318774E+02, 4.7657361737E+00, 9.9923084466E-01],
	[7.9500000000E+04, 1.2998647707E+02, 5.0600007243E+00, 2.4355516414E+02, 4.7749247500E+00, 9.9925616875E-01],
	[8.0000000000E+04, 1.3153676829E+02, 5.0657559384E+00, 2.4867110457E+02, 4.7840545425E+00, 9.9928054181E-01],
	[8.0500000000E+04, 1.3309571706E+02, 5.0714817791E+00, 2.5386158859E+02, 4.7930924090E+00, 9.9930400814E-01],
	[8.1000000000E+04, 1.3466536196E+02, 5.0772604764E+00, 2.5912726170E+02, 4.8021160834E+00, 9.9932661022E-01],
	[8.1500000000E+04, 1.3624369634E+02, 5.0830104730E+00, 2.6446873880E+02, 4.8110516920E+00, 9.9934837892E-01],
	[8.2000000000E+04, 1.3783153165E+02, 5.0887645027E+00, 2.6988663599E+02, 4.8199310674E+00, 9.9936934459E-01],
	[8.2500000000E+04, 1.3942927919E+02, 5.0945382388E+00, 2.7538157054E+02, 4.8287702782E+00, 9.9938953749E-01],
	[8.3000000000E+04, 1.4103572580E+02, 5.1002835798E+00, 2.8095416095E+02, 4.8375261034E+00, 9.9940898757E-01],
	[8.3500000000E+04, 1.4265244544E+02, 5.1060622914E+00, 2.8660502685E+02, 4.8462573421E+00, 9.9942772432E-01],
	[8.4000000000E+04, 1.4427835467E+02, 5.1118312865E+00, 2.9233478910E+02, 4.8549253211E+00, 9.9944577662E-01],
	[8.4500000000E+04, 1.4591317321E+02, 5.1175799154E+00, 2.9814406968E+02, 4.8635214061E+00, 9.9946317260E-01],
	[8.5000000000E+04, 1.4755856944E+02, 5.1233721680E+00, 3.0403349178E+02, 4.8721065217E+00, 9.9947993962E-01],
	[8.5500000000E+04, 1.4921270625E+02, 5.1291369465E+00, 3.1000367973E+02, 4.8806156431E+00, 9.9949610416E-01],
	[8.6000000000E+04, 1.5087634524E+02, 5.1349035854E+00, 3.1605525904E+02, 4.8890773150E+00, 9.9951169178E-01],
	[8.6500000000E+04, 1.5255001511E+02, 5.1406915329E+00, 3.2218885634E+02, 4.8975109029E+00, 9.9952672713E-01],
	[8.7000000000E+04, 1.5423244382E+02, 5.1464525880E+00, 3.2840509943E+02, 4.9058724877E+00, 9.9954123390E-01],
	[8.7500000000E+04, 1.5592487999E+02, 5.1522336977E+00, 3.3470461727E+02, 4.9142070723E+00, 9.9955523483E-01],
	[8.8000000000E+04, 1.5762688604E+02, 5.1580178261E+00, 3.4108803994E+02, 4.9224998144E+00, 9.9956875160E-01],
	[8.8500000000E+04, 1.5933766932E+02, 5.1637756214E+00, 3.4755599864E+02, 4.9307242333E+00, 9.9958180255E-01],
	[8.9000000000E+04, 1.6105887496E+02, 5.1695677596E+00, 3.5410912574E+02, 4.9389382236E+00, 9.9959440339E-01],
	[8.9500000000E+04, 1.6278927806E+02, 5.1753485261E+00, 3.6074805471E+02, 4.9470999143E+00, 9.9960656976E-01],
	[9.0000000000E+04, 1.6452847686E+02, 5.1811034979E+00, 3.6747342014E+02, 4.9551966805E+00, 9.9961831722E-01],
	[9.0500000000E+04, 1.6627841931E+02, 5.1869032995E+00, 3.7427555595E+02, 4.9632939972E+00, 9.9962966111E-01],
	[9.1000000000E+04, 1.6803727780E+02, 5.1926811183E+00, 3.8116508109E+02, 4.9713318665E+00, 9.9964061653E-01],
	[9.1500000000E+04, 1.6980505936E+02, 5.1984375635E+00, 3.8814262883E+02, 4.9793116909E+00, 9.9965119827E-01],
	[9.2000000000E+04, 1.7158358970E+02, 5.2042377819E+00, 3.9520883352E+02, 4.9872956571E+00, 9.9966142075E-01],
	[9.2500000000E+04, 1.7337095698E+02, 5.2100129819E+00, 4.0236433060E+02, 4.9952198259E+00, 9.9967129798E-01],
	[9.3000000000E+04, 1.7516737658E+02, 5.2157711713E+00, 4.0960975655E+02, 5.0030925061E+00, 9.9968084358E-01],
	[9.3500000000E+04, 1.7697444795E+02, 5.2215685731E+00, 4.1694574895E+02, 5.0109675141E+00, 9.9969007072E-01],
	[9.4000000000E+04, 1.7879037456E+02, 5.2273414477E+00, 4.2437294643E+02, 5.0187854893E+00, 9.9969899209E-01],
	[9.4500000000E+04, 1.8061538614E+02, 5.2330981899E+00, 4.3189198869E+02, 5.0265550583E+00, 9.9970761998E-01],
	[9.5000000000E+04, 1.8245104691E+02, 5.2388930661E+00, 4.3950351648E+02, 5.0343282193E+00, 9.9971596615E-01],
	[9.5500000000E+04, 1.8429558115E+02, 5.2446638894E+00, 4.4720817161E+02, 5.0420469148E+00, 9.9972404093E-01],
	[9.6000000000E+04, 1.8614913580E+02, 5.2504160870E+00, 4.5500659693E+02, 5.0497169214E+00, 9.9973185320E-01],
	[9.6500000000E+04, 1.8801343210E+02, 5.2562087096E+00, 4.6289943633E+02, 5.0573947903E+00, 9.9973941173E-01],
	[9.7000000000E+04, 1.8988662000E+02, 5.2619777373E+00, 4.7088733477E+02, 5.0650205818E+00, 9.9974672521E-01],
	[9.7500000000E+04, 1.9176869575E+02, 5.2677233873E+00, 4.7897093823E+02, 5.0725950919E+00, 9.9975380221E-01],
	[9.8000000000E+04, 1.9366164199E+02, 5.2735130061E+00, 4.8715089371E+02, 5.0801827667E+00, 9.9976065114E-01],
	[9.8500000000E+04, 1.9556352730E+02, 5.2792804769E+00, 4.9542784926E+02, 5.0877215395E+00, 9.9976728023E-01],
	[9.9000000000E+04, 1.9747431862E+02, 5.2850250050E+00, 5.0380245396E+02, 5.0952112025E+00, 9.9977369751E-01],
	[9.9500000000E+04, 1.9939570833E+02, 5.2908035114E+00, 5.1227535790E+02, 5.1027063478E+00, 9.9977991081E-01],
	[1.0000000000E+05, 2.0132633249E+02, 5.2965696473E+00, 5.2084721219E+02, 5.1101635409E+00, 9.9978592771E-01],
	[1.0050000000E+05, 2.0326588068E+02, 5.3023132616E+00, 5.2949298884E+02, 5.1175712516E+00, 9.9979175558E-01],
	[1.0100000000E+05, 2.0521565703E+02, 5.3080778329E+00, 5.3823830159E+02, 5.1249737965E+00, 9.9979740154E-01],
	[1.0150000000E+05, 2.0717506006E+02, 5.3138428415E+00, 5.4708379511E+02, 5.1323521940E+00, 9.9980287252E-01],
	[1.0200000000E+05, 2.0914340499E+02, 5.3195857360E+00, 5.5603011497E+02, 5.1396854724E+00, 9.9980817688E-01],
	[1.0250000000E+05, 2.1112151011E+02, 5.3253336301E+00, 5.6507790773E+02, 5.1469999047E+00, 9.9981332024E-01],
	[1.0300000000E+05, 2.1310973087E+02, 5.3310977053E+00, 5.7422782092E+02, 5.1543066257E+00, 9.9981830696E-01],
	[1.0350000000E+05, 2.1510691124E+02, 5.3368400609E+00, 5.8348050298E+02, 5.1615699921E+00, 9.9982314148E-01],
	[1.0400000000E+05, 2.1711328619E+02, 5.3425686131E+00, 5.9283660332E+02, 5.1687979712E+00, 9.9982782829E-01],
	[1.0450000000E+05, 2.1913036135E+02, 5.3483319350E+00, 6.0229677229E+02, 5.1760373192E+00, 9.9983237187E-01],
	[1.0500000000E+05, 2.2115641368E+02, 5.3540739196E+00, 6.1186166119E+02, 5.1832349654E+00, 9.9983677670E-01],
	[1.0550000000E+05, 2.2319144011E+02, 5.3597947470E+00, 6.2153192225E+02, 5.1903914708E+00, 9.9984104724E-01],
	[1.0600000000E+05, 2.2523696262E+02, 5.3655432755E+00, 6.3130820862E+02, 5.1975538773E+00, 9.9984518788E-01],
	[1.0650000000E+05, 2.2729192117E+02, 5.3712850445E+00, 6.4119117441E+02, 5.2046896941E+00, 9.9984920293E-01],
	[1.0700000000E+05, 2.2935587131E+02, 5.3770060210E+00, 6.5118147464E+02, 5.2117858872E+00, 9.9985309663E-01],
	[1.0750000000E+05, 2.3142954077E+02, 5.3827295192E+00, 6.6127976527E+02, 5.2188651000E+00, 9.9985687316E-01],
	[1.0800000000E+05, 2.3351343753E+02, 5.3884712159E+00, 6.7148670315E+02, 5.2259427023E+00, 9.9986053655E-01],
	[1.0850000000E+05, 2.3560634315E+02, 5.3941924743E+00, 6.8180294609E+02, 5.2329821062E+00, 9.9986409075E-01],
	[1.0900000000E+05, 2.3770825480E+02, 5.3998934606E+00, 6.9222915279E+02, 5.2399837989E+00, 9.9986753960E-01],
	[1.0950000000E+05, 2.3982096188E+02, 5.4056302612E+00, 7.0276598288E+02, 5.2470018045E+00, 9.9987088683E-01],
	[1.1000000000E+05, 2.4194285249E+02, 5.4113519231E+00, 7.1341409688E+02, 5.2539876957E+00, 9.9987413604E-01],
	[1.1050000000E+05, 2.4407376630E+02, 5.4170536514E+00, 7.2417415625E+02, 5.2609371886E+00, 9.9987729073E-01],
	[1.1100000000E+05, 2.4621448906E+02, 5.4227600536E+00, 7.3504682331E+02, 5.2678741671E+00, 9.9988035426E-01],
	[1.1150000000E+05, 2.4836539188E+02, 5.4284822307E+00, 7.4603276131E+02, 5.2748095957E+00, 9.9988332991E-01],
	[1.1200000000E+05, 2.5052533484E+02, 5.4341848027E+00, 7.5713263441E+02, 5.2817098627E+00, 9.9988622082E-01],
	[1.1250000000E+05, 2.5269431530E+02, 5.4398679240E+00, 7.6834710765E+02, 5.2885753944E+00, 9.9988903001E-01],
	[1.1300000000E+05, 2.5487395002E+02, 5.4455813052E+00, 7.7967684695E+02, 5.2954541716E+00, 9.9989176040E-01],
	[1.1350000000E+05, 2.5706294680E+02, 5.4512848137E+00, 7.9112251915E+02, 5.3023079839E+00, 9.9989441480E-01],
	[1.1400000000E+05, 2.5926099789E+02, 5.4569691861E+00, 8.0268479195E+02, 5.3091282041E+00, 9.9989699548E-01],
	[1.1450000000E+05, 2.6146851209E+02, 5.4626470997E+00, 8.1436433396E+02, 5.3159272634E+00, 9.9989950432E-01],
	[1.1500000000E+05, 2.6368658513E+02, 5.4683516285E+00, 8.2616181466E+02, 5.3227372051E+00, 9.9990194324E-01],
	[1.1550000000E+05, 2.6591372897E+02, 5.4740373270E+00, 8.3807790441E+02, 5.3295146343E+00, 9.9990431414E-01],
	[1.1600000000E+05, 2.6814994121E+02, 5.4797043386E+00, 8.5011327443E+02, 5.3362599256E+00, 9.9990661894E-01],
	[1.1650000000E+05, 2.7039622971E+02, 5.4853832350E+00, 8.6226859686E+02, 5.3430027125E+00, 9.9990885955E-01],
	[1.1700000000E+05, 2.7265248618E+02, 5.4910703267E+00, 8.7454454467E+02, 5.3497397033E+00, 9.9991103785E-01],
	[1.1750000000E+05, 2.7491782736E+02, 5.4967390250E+00, 8.8694179171E+02, 5.3564455576E+00, 9.9991315572E-01],
	[1.1800000000E+05, 2.7719225095E+02, 5.5023894670E+00, 8.9946101270E+02, 5.3631206242E+00, 9.9991521498E-01],
	[1.1850000000E+05, 2.7947724457E+02, 5.5080662089E+00, 9.1210288322E+02, 5.3698080144E+00, 9.9991721746E-01],
	[1.1900000000E+05, 2.8177173199E+02, 5.5137366453E+00, 9.2486807973E+02, 5.3764765756E+00, 9.9991916491E-01],
	[1.1950000000E+05, 2.8407531794E+02, 5.5193891071E+00, 9.3775727952E+02, 5.3831152795E+00, 9.9992105908E-01],
	[1.2000000000E+05, 2.8638800020E+02, 5.5250237262E+00, 9.5077116075E+02, 5.3897244515E+00, 9.9992290165E-01],
])