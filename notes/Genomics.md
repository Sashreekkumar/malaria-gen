

### Structure of a Genome:
- The genome of an organism is represented by chromosomes, which are sequences of DNA.
- Diploid organisms have two copies of each chromosome, one inherited from each parent.
- Each chromosome can be divided into two chromosome arms which are separated by the centromere, a constricted region of the chromosome involved in cell division.
- The _Anopheles gambiae_ genome for example is composed of three chromosome pairs each seperated into right (R) and left (L) chromosome arms.
- Two of these chromosome pairs (chromosomes 2 and 3) are autosomes, i.e. both males and females have two copies.
- The final chromosome pair, the X and Y chromosomes are those which determine sex, with two copies of the X chromosome found in females and one copy of X and one of Y found in males.    
- You may come across terms such as scaffolds or contigs, these are simply smaller units into which the genome sequence of a chromosome can be broken down

### Mutations in the Genome:
- **SNP (Single Nucleotide polymorphism):** A point mutation (single base substitution) where one nucleotide is different compared to the reference. In reality, their are millions of such SNPs in a single genome sequence, and these are called genetic markers. 

- Each cell contains around 37 millions cells and each cell contains the same DNA. The DNA contains chromosomes. Each cell contains 2

- When DNA is passed from one generation of mosquitoes to the next, it undergoes **mutations**, which are errors in the DNA copying process. There are different types of mutations that can occur. These include:
    - **Single Nucleotide Polymorphisms (SNPs)** - substitutions of a single letter in the DNA sequence        
    - **Copy Number Variants (CNVs)** - duplications or deletions of sections of a DNA sequence
- It is also very useful to know whether combinations of mutations occur together in the same DNA sequence. In order to reconstruct this information, another pipeline is used to produce **phased haplotypes**
- Time and place of collection is also equally important. 

- **Common SNP data type files:** 
	→ bed, bim, fam 
	→ fastq

### Phylogentic Tree:
- Phylogenetic trees are used to represent the evolutionary relationships(often called the **tips** or **leaves** of the tree) are connected together by tree **branches** with their length representing the amount of genetic change that has occured between individuals.
-  Individual branches are internally connected together by **parent nodes**. These nodes represent common ancestors inferred from the data. The common ancestor of all individuals in the tree is represented by its root.
- Such a tree with root is called rooted tree. A tree without such information called the unrooted tree can be designed, but we can’t infer the direction of the evolutionary path without applying time-scale based on known mutation rates.
- **Neighbour Joining Trees:** Based on minimum evolution; The concept assumes that the tree with the shortest sum of branch lenghts is correct tree i.e principle of parsimony. Neighbour-joining trees are derived by determining which individuals at the end of the tree branches are neighbours by clustering together the closest related individuals in a interative process.

### Transcription:
- Genes are transcribed into mRNA, which means that the DNA sequence from the gene region is “read” and used as a template to build an mRNA molecule with an equivalent nucleotide sequence. mRNA forms the template from which a protein's amino acids are formed.
- For many genes, the process of transcription is followed by splicing, which means that parts of the sequence are cut out before the mRNA is translated into protein. The parts of the gene sequence that are cut out are called “introns”. The parts of the gene sequence that are kept and included in the final mRNA molecule are called “exons”.
- Often there are alternative ways in which a gene sequence can be read during transcription. For example, each transcript may include different exons, or exons may vary in length. Alternative splicing can allow the same gene to encode different proteins, which may perform different functions.
- A change to the DNA sequence of a gene encoding for different transcripts has the potential to impact on protein function more than once, with the outcome dependent on the transcript under investigation.
- Mostly the exons comprise DNA sequence that is translated into an amino acid sequence, and these protein-coding segments are called “CDSs”, which stands for coding sequences
- Some genes also include “UTRs” which are untranslated regions at the start and end of the gene sequence. These UTRs are included in the exons but do not get translated into protein. A UTR at the start of the gene is called a “five prime UTR” and a UTR at the end of a gene is called a “three prime UTR”.
	**Exons:** Exons are the protein-coding sequences of DNA and RNA that remain in mature messenger RNA (mRNA) after RNA splicing, where non-coding introns are removed.
