// 模拟从book.json中获取的数据
// 异步加载book.json文件
fetch('book.json')
    .then(response => response.json())
    .then(data => {
        const books = data; // 假设book.json是一个数组

        // 将价格字符串转换为数字
        books.forEach(book => book.price = parseFloat(book.price.replace('¥', '')));

        // 根据价格区间对书籍进行分类
        const priceRanges = {
            '0-20': 0,
            '20-50': 0,
            '50-80': 0,
            '>80': 0
        };

        books.forEach(book => {
            const price = book.price;
            if (price <= 20) {
                priceRanges['0-20']++;
            } else if (price <= 50) {
                priceRanges['20-50']++;
            } else if (price <= 80) {
                priceRanges['50-80']++;
            } else {
          }