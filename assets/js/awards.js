if (window.matchMedia("(pointer: fine)").matches) {
        $("li.tag").on('mouseover',function(){
            $("li.tag").removeClass('active');//remove from other elements
            $(this).addClass('active');
        });
    } else {
        $("li.tag a").contents().unwrap();
        $("li.tag").on('click',function(){
            $("li.tag").removeClass('active');//remove from other elements
            $(this).addClass('active');
        });
    }