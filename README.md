# DETECTION OF ARCHAEOLOGICAL SITES FROM AERIAL IMAGERY USING DEEP LEARNING

**Documentation under construction.** 

You can ceck the results of the work here:


[https://lup.lub.lu.se/student-papers/search/publication/8974790](https://lup.lub.lu.se/student-papers/search/publication/8974790)

This work presents the results of an approach using 4 different Convolutional Neural Networks (CNN) models based on different architectures and learning methods. Of the models tested, 3 of them correspond to state of the art pre-trained models for which different techniques of transfer learning were used. The fourth one is a CNN architecture developed specifically for this task. The Deep Convolutional Neural Networks used were trained to carry a binary identification task, in this case, to determine whether an image contains any kind of topographical anomalies corresponding to archaeological structures,or not. The case studies were obtained from the southern Baltic sea region of Sweden and Birka and these correspond to aerial images in the visible light range and infrared. The kind of structures present on the images are burials of different shapes corresponding to the Viking ages.

By using the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) as measurement, the selection of the model best suitable for this task was carried out. Additionally, different augmentation techniques were tried including the generation of images using a Deep Convolutional Generative Adversarial Networks. Finally, an ensemble approach was tested combining the results obtained from the models which showed the best results individually in different types of airborne data. With this approach, a sensitivity of 76% with a specificity of 92% was achieved.
