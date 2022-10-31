# D'Advisor 

Программный модуль предназначен для упрощения поиска похожих по содержанию документов на определенном множестве документов. 


## 1.  Установка D'Advisor 
Программный модуль D’Advisor поставляется в форме лицензированной копии программного обеспечения  на любом электронном носителе и пакетом электронной документации, в которое входит: 
“Руководство по установке и эксплуатации модуля D’Advisor”


Установка ПО происходит путем копирования библиотеки в целевой каталог и запуском команды для установки пакета среду Python. Требуется подключение к интернету для загрузки зависимостей.

Перед установкой библиотеки проверьте, что у вас установлена ​​актуальная версия setuptools:

` python3 -m pip install --upgrade setuptools `

С помощью следующей команды выполните установку библиотеки D’Adviser:

`sudo python3 setup.py install `


## 2.  Обучение 

Перед использованием модуля “D’Adviser” необходимо обучить модель данных. Пример обучения програмного модуля: 

```Python
 from dadviser.core import DAdviser 
 documents_path = “/путь/к/директории/с/документами” 
 adviser = DAdviser(documents_path)
 ``` 



## 3.  Использование 

Пример  вызова программного модуля  из кода: 
```Python

from dadviser.core import DAdviser  
documents_path = “/путь/к/директории/с/документами” 
adviser = DAdviser(documents_path) 
compare_with_text = read_file(“/путь/к/файлу/для/проверки”) 
result = adviser.get_similarity(compare_with_text, top_list=10)
print(result)

 ``` 



## 4.  Программные и аппаратные требования к ПО
Рекомендуется устанавливать программный модуль на выделенный компьютер (сервер), отвечающий следующим техническими характеристикам: 
- Процессор Core i5 и выше (от 2.4 ГГц)
- Количество ядер CPU: 6 и выше 
- Размер оперативной памяти (RAM): 16 ГБ и выше (завист от количества текста)
Жесткий диск
-Тип памяти SSD/HDD 
- Размер свободного места не менее 100 ГБ (завист от количества текста) 

Поддерживаемые ОС:
- Microsoft Windows (64-bit)
- Fedora
- Debian Linux
- CentOS    
