# Anopeheles

![Data collection Process](images/flowchart_spec.png)

### Reference Genome:

### Voltage Gated Sodium Channel Gene
- One particular gene of interest is the voltage-gated sodium channel (*Vgsc*), also known as *para*, which encodes for a protein called VGSC which is the binding target of pyrethroid insecticides. 
- Pyrethroids function by binding to the associated protein and disrupting normal nervous system functions. 
- Changes to the nucleotide sequence of the *Vgsc* gene can alter the structure or function of proteins and prevent the 'knock-down' of the mosquito. As a result, an increased dose of insectide treatment may be required to cause death. 
- This is an example of how a change to a genotype can impact on phenotype.

Several nucleotide changes in the *Vgsc* gene have been implicated in an increased resistance to insecticides in *Anopheles* mosquitoes. These include the substitutions L995S found in Central and East Africa and the substitutions L995F and N1570Y which are present in Central and West Africa and often found together.

### Population Structure
- We expect that any two mosquitoes from the same species will be more closely related than two mosquitoes from different species.

- We might expect that two mosquitoes, of the same species, sample-+ from the same location, will be more closely related than two mosquitoes sampled from distant locations. These non-random patterns of relatedness are known as population structure.

- Mosquitoes may also be impeded from travelling in certain directions by natural physical barriers, such as a region of elevated terrain, or by variation in the availability of suitable habitat. These limits and barriers to movement can thus generate population structure, also known as **geographic isolation**.

- **Migration:** Also, recent studies have suggested that some malaria mosquitoes may engage in long-distance migration , challenging the previous view that mosquitoes generally don’t travel more than a few kilometres in their lifetimes. But we still don’t know to what extent long-distance migration occurs, or whether the rate or range of migration varies between geographical regions and/or mosquito species. 

- We would also like to learn more about how ecological and landscape variation affects the movement and interbreeding of mosquito populations in different regions of Africa.

### Phylogentic Trees
- Refer to [[Genomics]] for explanation
- Parameters required to create such a tree in MalariaGen are:
	1. **`region`** is one we are probably getting familar with by now, it defines what region of the genome we want to use to run the analysis. We could assign a whole chromosome arm, a chromosome region, with a start and stop point, or a specific gene of interest.
	2. **`n_snps`** requires us to define how many SNPs to use from our region to build the NJT. This reduction in SNPs is called thinning. Too few SNPs and the analysis will not have enought power to accurately build trees, too many SNPs and the analysis will take a long time to run. We generally find 100,000 SNPs is a good compromise when running on Google Colab.
	- `ag3.plot_njt?`
