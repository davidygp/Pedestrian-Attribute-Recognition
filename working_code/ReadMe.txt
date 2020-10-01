Citation.
Y. Deng, P. Luo, C. C. Loy, X. Tang, "Pedestrian attribute recognition at far distance", in Proceedings of ACM Multimedia (ACM MM), 2014



------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
                         PEdesTrian Attribute Dataset (PETA)
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------



------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
(1) Usage
This dataset is intended for research purposes only and as such cannot be used commercially. 
In addition, reference must be made to the aforementioned publications when this dataset is used 
in any academic and research reports.
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
(2) Attribute annotations
The pedestrian image labels are located at ./PETA dataset/$subset$/archive/Label.txt

The ID in the label file corresponds to the pedestrian ID as shown in the prefix of the image 
files in each subset. Although color ambiguity and internal noise (blurred) are inevitable, 
we have tried our best to label the pedestrian image with the following attributes

 1) 61 Binary Attributes
 2)  4 Multi-class attributes

The Binary Attributes are as follows.
(The names in the second column corresponds to the names used in our paper)

                    2	accessoryHeadphone	
                    4	personalLess15		
                    5	personalLess30		Age16-30
                    6	personalLess45		Age31-45
                    7	personalLess60		Age46-60
                    8	personalLarger60	AgeAbove60
                    9	carryingBabyBuggy	
                    10	carryingBackpack	Backpack
                    11	hairBald		
                    12	footwearBoots		
                    13	lowerBodyCapri		
                    14	carryingOther		CarryingOther
                    15	carryingShoppingTro	
                    16	carryingUmbrella	
                    17	lowerBodyCasual		Casual lower
                    18	upperBodyCasual		Casual upper
                    19	personalFemale		
                    20	carryingFolder		
                    21	lowerBodyFormal		Formal lower
                    22	upperBodyFormal		Formal upper
                    23	accessoryHairBand	
                    24	accessoryHat		Hat
                    25	lowerBodyHotPants	
                    26	upperBodyJacket		Jacket
                    27	lowerBodyJeans		Jeans
                    28	accessoryKerchief	
                    29	footwearLeatherShoes	Leather Shoes
                    30	upperBodyLogo		Logo
                    31	hairLong		Long hair
                    32	lowerBodyLongSkirt	
                    33	upperBodyLongSleeve	
                    35	lowerBodyPlaid		
                    37	lowerBodyThinStripes	
                    38	carryingLuggageCase	
                    39	personalMale		Male
                    40	carryingMessengerBag	MessengerBag
                    41	accessoryMuffler	Muffler
                    42	accessoryNothing	No accessory
                    43	carryingNothing		No carrying
                    44	upperBodyNoSleeve	
                    45	upperBodyPlaid		Plaid
                    46	carryingPlasticBags	Plastic bag
                    47	footwearSandals		Sandals
                    48	footwearShoes		Shoes
                    49	hairShort		
                    50	lowerBodyShorts		Shorts 
                    51	upperBodyShortSleeve	ShortSleeve
                    52	lowerBodyShortSkirt	Skirt
                    53	footwearSneaker		Sneaker
                    54	footwearStocking	
                    55	upperBodyThinStripes	Stripes
                    56	upperBodySuit		
                    57	carryingSuitcase	
                    58	lowerBodySuits		
                    59	accessorySunglasses	Sunglasses
                    60	upperBodySweater	
                    61	upperBodyThickStripes	
                    62	lowerBodyTrousers	Trousers
                    63	upperBodyTshirt		Tshirt
                    64	upperBodyOther		UpperOther
                    65	upperBodyVNeck		V-Neck


The multiclass attributes are as follows

1. footwear: 	Black, Blue, Brown, Green, Grey, Orange, Pink, Purple, Red, White, Yellow
2. hair:        Black, Blue, Brown, Green, Grey, Orange, Pink, Purple, Red, White, Yellow
3. lowerbody: 	Black, Blue, Brown, Green, Grey, Orange, Pink, Purple, Red, White, Yellow
4. upperbody:	Black, Blue, Brown, Green, Grey, Orange, Pink, Purple, Red, White, Yellow
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
Note that 

	1)in the PETA dataset, the following labeled binary attributes are discarded 
	  in the final composition of PETA since the number of sample images is too small 
	  ( < 10 ) 

		-- 1. accessoryFaceMask
		--34. lowerBodyLogo
		-- 3. accessoryShawl
		--36. lowerBodyThickStripes

	2)Some of the color attribtues may not have suffcient samples. 
	  (e.g. Green Hair.  GreeeeeeeenHair:D)

In brief, the PETA dataset contains 61 binary attributes and 4 multiclass attributes.
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
For more info, please visit the project page: http://mmlab.ie.cuhk.edu.hk/projects/PETA.html
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------

(updated on Oct 20, 2014)











