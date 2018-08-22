# Analyze-Mallet-results
Analyze-Mallet-results is an offline web application for visualization and analyze outputs of Mallet package.

### Required software
 * Python 2.7 interpreter
 * *[cherrypy]*, with *[jinja2]* templating
 * *[matplotlib]*
 * *[NumPy]*
 * *[Pandas]*
 * *[Pypi (python-bidi)]*
 * *[Seaborn]*
 * *[Sklearn]*
 * *[Spyre library](https://github.com/adamhajari/spyre)*
 


[cherrypy]:http://docs.cherrypy.org/en/latest/install.html
[jinja2]:http://jinja.pocoo.org/docs/dev/intro/#installation
[Pandas]:http://pandas.pydata.org/pandas-docs/stable/install.html#recommended-dependencies
[matplotlib]:http://matplotlib.org/users/installing.html
[Pypi (python-bidi)]:https://pypi.org/project/python-bidi/
[NumPy]:http://www.numpy.org/
[Sklearn]:http://scikit-learn.org/stable/
[Seaborn]:http://seaborn.pydata.org/index.html

### Before using Mallet
Analyze-Mallet-results work with results of topic-model on corpus of filse that divided in to different categories, and the analyze based on topic-model results and the diffrent categories.
the input data need to be in hierarchial path such as file of the properties "green" and "sweet" is in path like \path-to-input-files\green\sweet, or alternativly \path-to-input-files\sweet\green ׂׂ(Hierarchy does not matter).

Analyze-Mallet-results used output of Mallet package. 
The files Analyze-Mallet-results used is composition.txt and keys.text. these names is results of run the topic-model with the command
```bash
    $ bin \mallet train-topics your-own-parameters –optimaize-interval some-int –output-keys keys.txt –output-doc-topics composition.txt 
```
In addition, Analyze-Mallet-results use the files:
- size_of_files.txt files. this file is the results of the terminal command
```bash
    $ du –ch path\to\your\infput-for-Mallet-files > size_of_files.txt  
```
- labels_dic.txt
Analyze-Mallet-results assume that the input for Mallet files are divided into different categories. 
