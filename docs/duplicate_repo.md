# repo duplication (public to private) 


In order to add new functions/code on a public-available repository while keep the new content private, it is suggested that to make a mirror duplication thus the newly created one can act as private copy.



https://help.github.com/en/github/creating-cloning-and-archiving-repositories/duplicating-a-repository



## remove cached .pyc

```bash
> cd {path-of-repository-root}
> git rm --cached *.pyc
```