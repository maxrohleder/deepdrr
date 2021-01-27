import numpy as np

# data taken from McGPU spectra files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[MEAN FREE PATHS (cm)]
#[Energy (eV) 	| Rayleigh 	| Compton 	| Photoelectric 	| TOTAL (+pair prod) (cm) | Rayleigh: max cumul prob F^2]

air_MFP = np.array([
	[1.50000E+04, 6.3499630538E+03, 5.5098032495E+03, 6.2224742754E+02, 5.1386044196E+02, 6.8066554446E-01],
	[1.55000E+04, 6.6401620613E+03, 5.4700591825E+03, 6.9047487848E+02, 5.6126463763E+02, 6.9439619452E-01],
	[1.60000E+04, 6.9361827560E+03, 5.4332421170E+03, 7.6364786536E+02, 6.1060208893E+02, 7.0772770153E-01],
	[1.65000E+04, 7.2380917894E+03, 5.3990651935E+03, 8.4195255617E+02, 6.6177369335E+02, 7.2064806035E-01],
	[1.70000E+04, 7.5459196357E+03, 5.3672684431E+03, 9.2557601239E+02, 7.1467125016E+02, 7.3314775932E-01],
	[1.75000E+04, 7.8608483543E+03, 5.3381121514E+03, 1.0147061636E+03, 7.6920008942E+02, 7.4521997735E-01],
	[1.80000E+04, 8.1821679221E+03, 5.3110027568E+03, 1.1101860029E+03, 8.2558976899E+02, 7.5686066416E-01],
	[1.85000E+04, 8.5096632042E+03, 5.2856713202E+03, 1.2116510210E+03, 8.8337323287E+02, 7.6806744682E-01],
	[1.90000E+04, 8.8440244495E+03, 5.2622077969E+03, 1.3192984833E+03, 9.4243282219E+02, 7.7884092813E-01],
	[1.95000E+04, 9.1858862528E+03, 5.2406387764E+03, 1.4333265814E+03, 1.0026502078E+03, 7.8918377791E-01],
	[2.00000E+04, 9.5339585322E+03, 5.2203795311E+03, 1.5539344110E+03, 1.0638603624E+03, 7.9909975888E-01],
	[2.05000E+04, 9.8894310126E+03, 5.2016848749E+03, 1.6817426035E+03, 1.1261454405E+03, 8.0859519682E-01],
	[2.10000E+04, 1.0252402261E+04, 5.1844638172E+03, 1.8165880100E+03, 1.1891954794E+03, 8.1767723585E-01],
	[2.15000E+04, 1.0621354262E+04, 5.1681729591E+03, 1.9586757649E+03, 1.2528330425E+03, 8.2635441524E-01],
	[2.20000E+04, 1.0999049377E+04, 5.1535231550E+03, 2.1082118950E+03, 1.3170119010E+03, 8.3463654627E-01],
	[2.25000E+04, 1.1382837996E+04, 5.1396798062E+03, 2.2655975774E+03, 1.3815954930E+03, 8.4253633672E-01],
	[2.30000E+04, 1.1774220011E+04, 5.1269859913E+03, 2.4314056507E+03, 1.4466286444E+03, 8.5007747710E-01],
	[2.35000E+04, 1.2172835650E+04, 5.1152783377E+03, 2.6053773243E+03, 1.5117969864E+03, 8.5727244761E-01],
	[2.40000E+04, 1.2578183547E+04, 5.1043588925E+03, 2.7877272347E+03, 1.5769665062E+03, 8.6413225802E-01],
	[2.45000E+04, 1.2991580950E+04, 5.0945138357E+03, 2.9786709285E+03, 1.6420827453E+03, 8.7066807770E-01],
	[2.50000E+04, 1.3411160412E+04, 5.0852125736E+03, 3.1784248464E+03, 1.7069663590E+03, 8.7689149273E-01],
	[2.55000E+04, 1.3839252708E+04, 5.0769866760E+03, 3.3880949266E+03, 1.7718665702E+03, 8.8281414474E-01],
	[2.60000E+04, 1.4273305461E+04, 5.0691654647E+03, 3.6071052587E+03, 1.8363361760E+03, 8.8844780848E-01],
	[2.65000E+04, 1.4715947573E+04, 5.0623350702E+03, 3.8356793893E+03, 1.9004152141E+03, 8.9380418693E-01],
	[2.70000E+04, 1.5164669409E+04, 5.0558653505E+03, 4.0740417651E+03, 1.9639060796E+03, 8.9889494955E-01],
	[2.75000E+04, 1.5621679374E+04, 5.0502296187E+03, 4.3225191701E+03, 2.0268723689E+03, 9.0373155872E-01],
	[2.80000E+04, 1.6085241298E+04, 5.0449973461E+03, 4.5816096775E+03, 2.0892142483E+03, 9.0832534981E-01],
	[2.85000E+04, 1.6556415576E+04, 5.0403781171E+03, 4.8512167648E+03, 2.1508558075E+03, 9.1268733715E-01],
	[2.90000E+04, 1.7034968796E+04, 5.0362828502E+03, 5.1315708221E+03, 2.2117335741E+03, 9.1682843703E-01],
	[2.95000E+04, 1.7520087385E+04, 5.0325216240E+03, 5.4229031193E+03, 2.2717568426E+03, 9.2075895698E-01],
	[3.00000E+04, 1.8013762131E+04, 5.0294743288E+03, 5.7254457932E+03, 2.3310026729E+03, 9.2448896939E-01],
	[3.05000E+04, 1.8512842081E+04, 5.0264792042E+03, 6.0405059175E+03, 2.3894176533E+03, 9.2802853438E-01],
	[3.10000E+04, 1.9021464290E+04, 5.0243478013E+03, 6.3673405949E+03, 2.4470553023E+03, 9.3138708498E-01],
	[3.15000E+04, 1.9535842485E+04, 5.0222999965E+03, 6.7061885284E+03, 2.5036558168E+03, 9.3457354464E-01],
	[3.20000E+04, 2.0057857801E+04, 5.0207001568E+03, 7.0572892967E+03, 2.5593209332E+03, 9.3759677356E-01],
	[3.25000E+04, 2.0587533166E+04, 5.0195246227E+03, 7.4208833422E+03, 2.6140374827E+03, 9.4046523281E-01],
	[3.30000E+04, 2.1122788376E+04, 5.0183673053E+03, 7.7972119607E+03, 2.6676540430E+03, 9.4318677973E-01],
	[3.35000E+04, 2.1667629990E+04, 5.0179771087E+03, 8.1865172899E+03, 2.7204418070E+03, 9.4576912442E-01],
	[3.40000E+04, 2.2218244943E+04, 5.0176116050E+03, 8.5890422995E+03, 2.7721297192E+03, 9.4821966298E-01],
	[3.45000E+04, 2.2775852457E+04, 5.0175004953E+03, 9.0050307809E+03, 2.8228045732E+03, 9.5054524394E-01],
	[3.50000E+04, 2.3341765818E+04, 5.0178673260E+03, 9.4347273371E+03, 2.8725651147E+03, 9.5275252688E-01],
	[3.55000E+04, 2.3913413682E+04, 5.0182289795E+03, 9.8804597618E+03, 2.9214063762E+03, 9.5484787646E-01],
	[3.60000E+04, 2.4493084995E+04, 5.0189989276E+03, 1.0340557553E+04, 2.9693207947E+03, 9.5683719833E-01],
	[3.65000E+04, 2.5080068864E+04, 5.0200285458E+03, 1.0815273834E+04, 3.0162620349E+03, 9.5872621090E-01],
	[3.70000E+04, 2.5672846266E+04, 5.0210443619E+03, 1.1304862581E+04, 3.0621219584E+03, 9.6052033196E-01],
	[3.75000E+04, 2.6274233652E+04, 5.0225329749E+03, 1.1809578620E+04, 3.1071354034E+03, 9.6222461703E-01],
	[3.80000E+04, 2.6882397620E+04, 5.0241588430E+03, 1.2329677609E+04, 3.1511697023E+03, 9.6384397062E-01],
	[3.85000E+04, 2.7496412335E+04, 5.0257639733E+03, 1.2865416035E+04, 3.1941649766E+03, 9.6538301752E-01],
	[3.90000E+04, 2.8119170588E+04, 5.0278292279E+03, 1.3417051207E+04, 3.2363722232E+03, 9.6684595703E-01],
	[3.95000E+04, 2.8748598820E+04, 5.0299929715E+03, 1.3984841239E+04, 3.2776382328E+03, 9.6823685510E-01],
	[4.00000E+04, 2.9383932803E+04, 5.0321304111E+03, 1.4569045053E+04, 3.3179174877E+03, 9.6955964298E-01],
	[4.05000E+04, 3.0027713388E+04, 5.0346487397E+03, 1.5170689546E+04, 3.3574765037E+03, 9.7081809448E-01],
	[4.10000E+04, 3.0678479155E+04, 5.0372990573E+03, 1.5789320623E+04, 3.3961718833E+03, 9.7201567990E-01],
	[4.15000E+04, 3.1335203561E+04, 5.0399186190E+03, 1.6425201226E+04, 3.4339368576E+03, 9.7315556741E-01],
	[4.20000E+04, 3.1999678581E+04, 5.0427826297E+03, 1.7078595087E+04, 3.4709404699E+03, 9.7424081616E-01],
	[4.25000E+04, 3.2671881151E+04, 5.0458745792E+03, 1.7749766721E+04, 3.5071988918E+03, 9.7527437388E-01],
	[4.30000E+04, 3.3350094273E+04, 5.0489322274E+03, 1.8438981412E+04, 3.5425876877E+03, 9.7625906157E-01],
	[4.35000E+04, 3.4034935191E+04, 5.0520482559E+03, 1.9146505217E+04, 3.5771782941E+03, 9.7719747126E-01],
	[4.40000E+04, 3.4728665422E+04, 5.0555418357E+03, 1.9872604948E+04, 3.6111783505E+03, 9.7809199594E-01],
	[4.45000E+04, 3.5428457121E+04, 5.0589983142E+03, 2.0617548172E+04, 3.6443698658E+03, 9.7894494038E-01],
	[4.50000E+04, 3.6134294251E+04, 5.0624184958E+03, 2.1381603201E+04, 3.6767711789E+03, 9.7975851770E-01],
	[4.55000E+04, 3.6848671195E+04, 5.0661443177E+03, 2.2167059921E+04, 3.7086654424E+03, 9.8053482851E-01],
	[4.60000E+04, 3.7570127111E+04, 5.0699648417E+03, 2.2972295864E+04, 3.7398890287E+03, 9.8127578818E-01],
	[4.65000E+04, 3.8297679440E+04, 5.0737468966E+03, 2.3797585422E+04, 3.7703791858E+03, 9.8198320673E-01],
	[4.70000E+04, 3.9031723478E+04, 5.0775442774E+03, 2.4643203755E+04, 3.8001873044E+03, 9.8265882386E-01],
	[4.75000E+04, 3.9774955437E+04, 5.0816989390E+03, 2.5509426778E+04, 3.8295509044E+03, 9.8330430350E-01],
	[4.80000E+04, 4.0524334710E+04, 5.0858134417E+03, 2.6396531161E+04, 3.8582378996E+03, 9.8392118983E-01],
	[4.85000E+04, 4.1279847314E+04, 5.0898885862E+03, 2.7304794317E+04, 3.8862652543E+03, 9.8451091498E-01],
	[4.90000E+04, 4.2042790914E+04, 5.0940835056E+03, 2.8234494400E+04, 3.9137545329E+03, 9.8507485318E-01],
	[4.95000E+04, 4.2814106150E+04, 5.0985040483E+03, 2.9185910297E+04, 3.9407957105E+03, 9.8561432226E-01],
	[5.00000E+04, 4.3591606393E+04, 5.1028839411E+03, 3.0159321624E+04, 3.9672297860E+03, 9.8613059162E-01],
	[5.05000E+04, 4.4375278645E+04, 5.1072239587E+03, 3.1156745108E+04, 3.9931009586E+03, 9.8662478630E-01],
	[5.10000E+04, 4.5166813761E+04, 5.1117172543E+03, 3.2176820398E+04, 4.0185270869E+03, 9.8709796477E-01],
	[5.15000E+04, 4.5966413912E+04, 5.1163788999E+03, 3.3219831661E+04, 4.0435368131E+03, 9.8755114792E-01],
	[5.20000E+04, 4.6772238584E+04, 5.1209996944E+03, 3.4286063792E+04, 4.0680016029E+03, 9.8798531681E-01],
	[5.25000E+04, 4.7584275683E+04, 5.1255803837E+03, 3.5375802405E+04, 4.0919357684E+03, 9.8840141125E-01],
	[5.30000E+04, 4.8404064702E+04, 5.1302855227E+03, 3.6489333832E+04, 4.1154699489E+03, 9.8880032912E-01],
	[5.35000E+04, 4.9232164073E+04, 5.1351672708E+03, 3.7626945113E+04, 4.1386574899E+03, 9.8918292572E-01],
	[5.40000E+04, 5.0066529442E+04, 5.1400081872E+03, 3.8788923996E+04, 4.1613574997E+03, 9.8954998337E-01],
	[5.45000E+04, 5.0907149546E+04, 5.1448089869E+03, 3.9975558926E+04, 4.1835828600E+03, 9.8990221093E-01],
	[5.50000E+04, 5.1754814350E+04, 5.1496496398E+03, 4.1187139046E+04, 4.2054042968E+03, 9.9024029082E-01],
	[5.55000E+04, 5.2611623357E+04, 5.1547328308E+03, 4.2423954188E+04, 4.2269844908E+03, 9.9056488068E-01],
	[5.60000E+04, 5.3474741534E+04, 5.1597753823E+03, 4.3686294869E+04, 4.2481284621E+03, 9.9087661235E-01],
	[5.65000E+04, 5.4344158386E+04, 5.1647779778E+03, 4.4974452290E+04, 4.2688477184E+03, 9.9117609144E-01],
	[5.70000E+04, 5.5219863528E+04, 5.1697412832E+03, 4.6288718324E+04, 4.2891534694E+03, 9.9146389712E-01],
	[5.75000E+04, 5.6104994072E+04, 5.1749541256E+03, 4.7629385517E+04, 4.3092750279E+03, 9.9174057457E-01],
	[5.80000E+04, 5.6997070790E+04, 5.1801817111E+03, 4.8996747085E+04, 4.3290474199E+03, 9.9200661447E-01],
	[5.85000E+04, 5.7895491497E+04, 5.1853696371E+03, 5.0391096902E+04, 4.3484394033E+03, 9.9226248322E-01],
	[5.90000E+04, 5.8800246507E+04, 5.1905185394E+03, 5.1812729503E+04, 4.3674609309E+03, 9.9250863104E-01],
	[5.95000E+04, 5.9712410679E+04, 5.1957225639E+03, 5.3261940076E+04, 4.3861941887E+03, 9.9274549139E-01],
	[6.00000E+04, 6.0633644556E+04, 5.2011203271E+03, 5.4739024459E+04, 4.4047572855E+03, 9.9297348064E-01],
	[6.05000E+04, 6.1561268875E+04, 5.2064788331E+03, 5.6248209435E+04, 4.4230034532E+03, 9.9319299799E-01],
	[6.10000E+04, 6.2495274602E+04, 5.2117986884E+03, 5.7786048624E+04, 4.4409161275E+03, 9.9340441946E-01],
	[6.15000E+04, 6.3435652790E+04, 5.2170804851E+03, 5.9352844005E+04, 4.4585039152E+03, 9.9360808602E-01],
	[6.20000E+04, 6.4384522614E+04, 5.2224962594E+03, 6.0948898223E+04, 4.4759114078E+03, 9.9380432344E-01],
	[6.25000E+04, 6.5341542998E+04, 5.2280131690E+03, 6.2574514592E+04, 4.4931219332E+03, 9.9399344649E-01],
	[6.30000E+04, 6.6304992798E+04, 5.2334918771E+03, 6.4229997083E+04, 4.5100326707E+03, 9.9417575866E-01],
	[6.35000E+04, 6.7274863664E+04, 5.2389329484E+03, 6.5915650327E+04, 4.5266512245E+03, 9.9435155190E-01],
	[6.40000E+04, 6.8251147327E+04, 5.2443369346E+03, 6.7631779605E+04, 4.5429849844E+03, 9.9452110939E-01],
	[6.45000E+04, 6.9236299762E+04, 5.2498907773E+03, 6.9378690848E+04, 4.5591923970E+03, 9.9468470738E-01],
	[6.50000E+04, 7.0229378668E+04, 5.2555179463E+03, 7.1156690634E+04, 4.5752194566E+03, 9.9484257779E-01],
	[6.55000E+04, 7.1228927934E+04, 5.2611079571E+03, 7.2966086179E+04, 4.5909831482E+03, 9.9499494185E-01],
	[6.60000E+04, 7.2234939840E+04, 5.2666613361E+03, 7.4807185340E+04, 4.6064899967E+03, 9.9514201515E-01],
	[6.65000E+04, 7.3247406738E+04, 5.2721785981E+03, 7.6680296605E+04, 4.6217463418E+03, 9.9528400721E-01],
	[6.70000E+04, 7.4268343491E+04, 5.2778042703E+03, 7.8585729096E+04, 4.6368773945E+03, 9.9542112107E-01],
	[6.75000E+04, 7.5297747512E+04, 5.2835337482E+03, 8.0523792558E+04, 4.6518861496E+03, 9.9555355307E-01],
	[6.80000E+04, 7.6333664504E+04, 5.2892270929E+03, 8.2494797363E+04, 4.6666626925E+03, 9.9568149269E-01],
	[6.85000E+04, 7.7376087330E+04, 5.2948847954E+03, 8.4499054501E+04, 4.6812126230E+03, 9.9580512248E-01],
	[6.90000E+04, 7.8425008914E+04, 5.3005073366E+03, 8.6536875577E+04, 4.6955413829E+03, 9.9592461805E-01],
	[6.95000E+04, 7.9481158891E+04, 5.3061447463E+03, 8.8608572814E+04, 4.7096958906E+03, 9.9604014810E-01],
	[7.00000E+04, 8.0547144262E+04, 5.3119693988E+03, 9.0714459039E+04, 4.7238265147E+03, 9.9615187433E-01],
	[7.05000E+04, 8.1619686543E+04, 5.3177589201E+03, 9.2855117205E+04, 4.7377523015E+03, 9.9625994279E-01],
	[7.10000E+04, 8.2698779136E+04, 5.3235137689E+03, 9.5030603006E+04, 4.7514773339E+03, 9.9636448351E-01],
	[7.15000E+04, 8.3784415492E+04, 5.3292343949E+03, 9.7241231302E+04, 4.7650062905E+03, 9.9646562320E-01],
	[7.20000E+04, 8.4876589117E+04, 5.3349212388E+03, 9.9487317553E+04, 4.7783437191E+03, 9.9656348563E-01],
	[7.25000E+04, 8.5977935716E+04, 5.3407424289E+03, 1.0176917782E+05, 4.7916372326E+03, 9.9665819142E-01],
	[7.30000E+04, 8.7087322574E+04, 5.3466218721E+03, 1.0408712874E+05, 4.8048267525E+03, 9.9674985782E-01],
	[7.35000E+04, 8.8203304712E+04, 5.3524675886E+03, 1.0644148755E+05, 4.8178377293E+03, 9.9683859856E-01],
	[7.40000E+04, 8.9325876073E+04, 5.3582799994E+03, 1.0883257209E+05, 4.8306741929E+03, 9.9692452376E-01],
	[7.45000E+04, 9.0455030645E+04, 5.3640595173E+03, 1.1126070075E+05, 4.8433400622E+03, 9.9700773989E-01],
	[7.50000E+04, 9.1590762466E+04, 5.3698065474E+03, 1.1372619253E+05, 4.8558391479E+03, 9.9708834972E-01],
	[7.55000E+04, 9.2736725239E+04, 5.3757422321E+03, 1.1622936698E+05, 4.8683662849E+03, 9.9716645233E-01],
	[7.60000E+04, 9.3889788569E+04, 5.3816731907E+03, 1.1877054425E+05, 4.8807582351E+03, 9.9724213980E-01],
	[7.65000E+04, 9.5049486555E+04, 5.3875717368E+03, 1.2135004505E+05, 4.8929942218E+03, 9.9731549352E-01],
	[7.70000E+04, 9.6215813635E+04, 5.3934382576E+03, 1.2396819065E+05, 4.9050776304E+03, 9.9738659224E-01],
	[7.75000E+04, 9.7388764286E+04, 5.3992731333E+03, 1.2662530290E+05, 4.9170117552E+03, 9.9745551306E-01],
	[7.80000E+04, 9.8568333027E+04, 5.4050767370E+03, 1.2932170421E+05, 4.9287998014E+03, 9.9752233132E-01],
	[7.85000E+04, 9.9758241782E+04, 5.4110638452E+03, 1.3205771755E+05, 4.9406327806E+03, 9.9758712043E-01],
	[7.90000E+04, 1.0095524182E+05, 5.4170433346E+03, 1.3483366644E+05, 4.9523467221E+03, 9.9764995181E-01],
	[7.95000E+04, 1.0215891657E+05, 5.4229916434E+03, 1.3764987496E+05, 4.9639236359E+03, 9.9771089482E-01],
	[8.00000E+04, 1.0336926092E+05, 5.4289091289E+03, 1.4050666776E+05, 4.9753663777E+03, 9.9777001668E-01],
	[8.05000E+04, 1.0458626981E+05, 5.4347961420E+03, 1.4340561592E+05, 4.9866792346E+03, 9.9782738249E-01],
	[8.10000E+04, 1.0580993818E+05, 5.4406530272E+03, 1.4634584431E+05, 4.9978633530E+03, 9.9788305858E-01],
	[8.15000E+04, 1.0704305695E+05, 5.4466339226E+03, 1.4932767998E+05, 5.0090575732E+03, 9.9793711047E-01],
	[8.20000E+04, 1.0828424011E+05, 5.4526591954E+03, 1.5235145053E+05, 5.0201942451E+03, 9.9798958606E-01],
	[8.25000E+04, 1.0953213848E+05, 5.4586544449E+03, 1.5541748413E+05, 5.0312097876E+03, 9.9804053202E-01],
	[8.30000E+04, 1.1078674735E+05, 5.4646200013E+03, 1.5852610945E+05, 5.0421066217E+03, 9.9808999469E-01],
	[8.35000E+04, 1.1204806207E+05, 5.4705561892E+03, 1.6167765573E+05, 5.0528871065E+03, 9.9813801987E-01],
	[8.40000E+04, 1.1331607800E+05, 5.4764633278E+03, 1.6487245275E+05, 5.0635535402E+03, 9.9818465273E-01],
	[8.45000E+04, 1.1459161574E+05, 5.4823852429E+03, 1.6811083080E+05, 5.0741470539E+03, 9.9822993763E-01],
	[8.50000E+04, 1.1587721494E+05, 5.4884537483E+03, 1.7139312072E+05, 5.0847877367E+03, 9.9827391809E-01],
	[8.55000E+04, 1.1716957006E+05, 5.4944933194E+03, 1.7471965389E+05, 5.0953207901E+03, 9.9831663667E-01],
	[8.60000E+04, 1.1846867678E+05, 5.5005042622E+03, 1.7809076219E+05, 5.1057482747E+03, 9.9835813490E-01],
	[8.65000E+04, 1.1977453083E+05, 5.5064868776E+03, 1.8150677805E+05, 5.1160721998E+03, 9.9839845327E-01],
	[8.70000E+04, 1.2108712792E+05, 5.5124414617E+03, 1.8496803441E+05, 5.1262945245E+03, 9.9843763114E-01],
	[8.75000E+04, 1.2240646384E+05, 5.5183683058E+03, 1.8847486475E+05, 5.1364171594E+03, 9.9847570678E-01],
	[8.80000E+04, 1.2373448005E+05, 5.5243653788E+03, 1.9202760304E+05, 5.1465301114E+03, 9.9851271730E-01],
	[8.85000E+04, 1.2507150784E+05, 5.5304468296E+03, 1.9562658378E+05, 5.1566480883E+03, 9.9854869867E-01],
	[8.90000E+04, 1.2641532829E+05, 5.5365006566E+03, 1.9927214201E+05, 5.1666716722E+03, 9.9858368573E-01],
	[8.95000E+04, 1.2776593748E+05, 5.5425271397E+03, 2.0296461323E+05, 5.1766025827E+03, 9.9861771215E-01],
	[9.00000E+04, 1.2912333148E+05, 5.5485265539E+03, 2.0670433351E+05, 5.1864424978E+03, 9.9865081051E-01],
	[9.05000E+04, 1.3048750640E+05, 5.5544991703E+03, 2.1048998408E+05, 5.1961920463E+03, 9.9868301169E-01],
	[9.10000E+04, 1.3185845836E+05, 5.5604452556E+03, 2.1432350702E+05, 5.2058538700E+03, 9.9871434120E-01],
	[9.15000E+04, 1.3323812207E+05, 5.5664584264E+03, 2.1820523916E+05, 5.2155144520E+03, 9.9874482263E-01],
	[9.20000E+04, 1.3462684043E+05, 5.5725528982E+03, 2.2213551787E+05, 5.2251883312E+03, 9.9877447952E-01],
	[9.25000E+04, 1.3602238846E+05, 5.5786209559E+03, 2.2611468099E+05, 5.2347788794E+03, 9.9880333520E-01],
	[9.30000E+04, 1.3742476257E+05, 5.5846628558E+03, 2.3014306684E+05, 5.2442875395E+03, 9.9883141275E-01],
	[9.35000E+04, 1.3883395917E+05, 5.5906788504E+03, 2.3422101427E+05, 5.2537157209E+03, 9.9885873492E-01],
	[9.40000E+04, 1.4024997470E+05, 5.5966691884E+03, 2.3834886260E+05, 5.2630648002E+03, 9.9888532412E-01],
	[9.45000E+04, 1.4167280563E+05, 5.6026341149E+03, 2.4252695163E+05, 5.2723361221E+03, 9.9891120231E-01],
	[9.50000E+04, 1.4310320757E+05, 5.6086092687E+03, 2.4675562165E+05, 5.2815634243E+03, 9.9893639105E-01],
	[9.55000E+04, 1.4454383735E+05, 5.6147168891E+03, 2.5103521347E+05, 5.2908599641E+03, 9.9896091137E-01],
	[9.60000E+04, 1.4599133349E+05, 5.6207992148E+03, 2.5536606832E+05, 5.3000824178E+03, 9.9898478386E-01],
	[9.65000E+04, 1.4744569269E+05, 5.6268564815E+03, 2.5974852797E+05, 5.3092320051E+03, 9.9900802857E-01],
	[9.70000E+04, 1.4890691169E+05, 5.6328889215E+03, 2.6418293463E+05, 5.3183099179E+03, 9.9903066503E-01],
	[9.75000E+04, 1.5037498724E+05, 5.6388967636E+03, 2.6866963100E+05, 5.3273173218E+03, 9.9905271224E-01],
	[9.80000E+04, 1.5184991608E+05, 5.6448802337E+03, 2.7320896027E+05, 5.3362553560E+03, 9.9907418867E-01],
	[9.85000E+04, 1.5333169501E+05, 5.6508395541E+03, 2.7780126607E+05, 5.3451251349E+03, 9.9909511224E-01],
	[9.90000E+04, 1.5482284339E+05, 5.6568883505E+03, 2.8244689253E+05, 5.3540323530E+03, 9.9911550025E-01],
	[9.95000E+04, 1.5632248202E+05, 5.6629848909E+03, 2.8714618424E+05, 5.3629396380E+03, 9.9913536778E-01],
	[1.00000E+05, 1.5782902011E+05, 5.6690573930E+03, 2.9189948627E+05, 5.3717816776E+03, 9.9915472823E-01],
	[1.00500E+05, 1.5934245465E+05, 5.6751060713E+03, 2.9668762690E+05, 5.3805530676E+03, 9.9917359496E-01],
	[1.01000E+05, 1.6086278268E+05, 5.6811311369E+03, 3.0152993652E+05, 5.3892614189E+03, 9.9919198119E-01],
	[1.01500E+05, 1.6239000123E+05, 5.6871327983E+03, 3.0642675485E+05, 5.3979076970E+03, 9.9920990000E-01],
	[1.02000E+05, 1.6392410737E+05, 5.6931112610E+03, 3.1137842201E+05, 5.4064928468E+03, 9.9922736431E-01],
	[1.02500E+05, 1.6546509815E+05, 5.6990667277E+03, 3.1638527857E+05, 5.4150177935E+03, 9.9924438682E-01],
	[1.03000E+05, 1.6701599332E+05, 5.7051307381E+03, 3.2144766553E+05, 5.4236053272E+03, 9.9926098004E-01],
	[1.03500E+05, 1.6857488506E+05, 5.7112181678E+03, 3.2656592431E+05, 5.4321773875E+03, 9.9927715621E-01],
	[1.04000E+05, 1.7014070918E+05, 5.7172827089E+03, 3.3174039676E+05, 5.4406917156E+03, 9.9929292734E-01],
	[1.04500E+05, 1.7171346295E+05, 5.7233245567E+03, 3.3697142516E+05, 5.4491491573E+03, 9.9930830518E-01],
	[1.05000E+05, 1.7329314366E+05, 5.7293439043E+03, 3.4225935220E+05, 5.4575505413E+03, 9.9932330119E-01],
	[1.05500E+05, 1.7487974861E+05, 5.7353409419E+03, 3.4760452100E+05, 5.4658966788E+03, 9.9933792656E-01],
	[1.06000E+05, 1.7647327514E+05, 5.7413158573E+03, 3.5300727510E+05, 5.4741883649E+03, 9.9935219221E-01],
	[1.06500E+05, 1.7807372055E+05, 5.7472688361E+03, 3.5846795846E+05, 5.4824263783E+03, 9.9936610876E-01],
	[1.07000E+05, 1.7968331572E+05, 5.7532941444E+03, 3.6398691545E+05, 5.4906992585E+03, 9.9937968643E-01],
	[1.07500E+05, 1.8130168271E+05, 5.7593740550E+03, 3.6956449086E+05, 5.4989912229E+03, 9.9939293415E-01],
	[1.08000E+05, 1.8292701462E+05, 5.7654321328E+03, 3.7520102989E+05, 5.5072315730E+03, 9.9940586012E-01],
	[1.08500E+05, 1.8455930896E+05, 5.7714685567E+03, 3.8089687815E+05, 5.5154210236E+03, 9.9941847251E-01],
	[1.09000E+05, 1.8619856327E+05, 5.7774835036E+03, 3.8665238168E+05, 5.5235602756E+03, 9.9943077938E-01],
	[1.09500E+05, 1.8784477509E+05, 5.7834771477E+03, 3.9246788691E+05, 5.5316500161E+03, 9.9944278872E-01],
	[1.10000E+05, 1.8949794198E+05, 5.7894496612E+03, 3.9834374069E+05, 5.5396909186E+03, 9.9945450842E-01],
	[1.10500E+05, 1.9115806151E+05, 5.7954012143E+03, 4.0428029026E+05, 5.5476836437E+03, 9.9946594623E-01],
	[1.11000E+05, 1.9282528154E+05, 5.8013381383E+03, 4.1027788328E+05, 5.5556346162E+03, 9.9947710976E-01],
	[1.11500E+05, 1.9450331912E+05, 5.8074118536E+03, 4.1633686781E+05, 5.5636863483E+03, 9.9948800650E-01],
	[1.12000E+05, 1.9618835354E+05, 5.8134647096E+03, 4.2245759231E+05, 5.5716916304E+03, 9.9949864378E-01],
	[1.12500E+05, 1.9788038253E+05, 5.8194968707E+03, 4.2864040566E+05, 5.5796510712E+03, 9.9950902877E-01],
	[1.13000E+05, 1.9957940386E+05, 5.8255084989E+03, 4.3488565710E+05, 5.5875652677E+03, 9.9951916849E-01],
	[1.13500E+05, 2.0128541530E+05, 5.8314997548E+03, 4.4119369631E+05, 5.5954348059E+03, 9.9952906980E-01],
	[1.14000E+05, 2.0299841460E+05, 5.8374707964E+03, 4.4756487335E+05, 5.6032602606E+03, 9.9953873937E-01],
	[1.14500E+05, 2.0471839957E+05, 5.8434217804E+03, 4.5399953866E+05, 5.6110421959E+03, 9.9954818475E-01],
	[1.15000E+05, 2.0644536799E+05, 5.8493528612E+03, 4.6049804311E+05, 5.6187811656E+03, 9.9955741306E-01],
	[1.15500E+05, 2.0818004452E+05, 5.8552931334E+03, 4.6706073792E+05, 5.6265049685E+03, 9.9956642891E-01],
	[1.16000E+05, 2.0992494851E+05, 5.8613417873E+03, 4.7368797475E+05, 5.6343075045E+03, 9.9957523689E-01],
	[1.16500E+05, 2.1167687839E+05, 5.8673706336E+03, 4.8038010560E+05, 5.6420685034E+03, 9.9958384163E-01],
	[1.17000E+05, 2.1343583211E+05, 5.8733798217E+03, 4.8713748291E+05, 5.6497884774E+03, 9.9959224775E-01],
	[1.17500E+05, 2.1520180763E+05, 5.8793694992E+03, 4.9396045946E+05, 5.6574679294E+03, 9.9960045989E-01],
	[1.18000E+05, 2.1697480294E+05, 5.8853398122E+03, 5.0084938846E+05, 5.6651073534E+03, 9.9960848264E-01],
	[1.18500E+05, 2.1875481601E+05, 5.8912909050E+03, 5.0780462348E+05, 5.6727072345E+03, 9.9961632059E-01],
	[1.19000E+05, 2.2054184483E+05, 5.8972229201E+03, 5.1482651847E+05, 5.6802680493E+03, 9.9962397827E-01],
	[1.19500E+05, 2.2233588740E+05, 5.9031359984E+03, 5.2191542778E+05, 5.6877902657E+03, 9.9963146017E-01],
	[1.20000E+05, 2.2413694173E+05, 5.9090302793E+03, 5.2907170613E+05, 5.6952743435E+03, 9.9963877071E-01],
])